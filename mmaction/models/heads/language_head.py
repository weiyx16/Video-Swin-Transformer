import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import normal_init
from mmaction.utils import get_root_logger

from ..builder import HEADS
from .base import BaseHead
from ..diffdist.functional import all_gather

import random
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    out_list = all_gather(out_list, x)  # if use dist.all_gather, you can't get gradient over each

    return torch.cat(out_list, dim=0)

def dist_collect_list(x, device):
    x = torch.tensor(x).to(device)
    x = dist_collect(x)
    return x.cpu().tolist()

def varsize_dist_collect(tensor: torch.Tensor):
    # ref: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
    # ref: https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
    tensor = tensor.contiguous()

    size_tens = dist_collect_list([tensor.shape[0]], tensor.device)
    max_size = max(size_tens)

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    ag = dist_collect(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)

class SyncFunction(torch.autograd.Function):
    # from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py 
    # support to do global sim matrix or local sim matrix
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class LModel(nn.Module):
    def __init__(self):
        super(LModel, self).__init__()
        # https://huggingface.co/roberta-base; config: https://huggingface.co/roberta-base/resolve/main/config.json
        # downloaded to /root/.cache/huggingface/transformers
        configuration = RobertaConfig.from_pretrained('roberta-base')
        configuration = configuration.__dict__.copy()
        configuration.update({'return_dict': False})
        configuration.update({'gradient_checkpointing': False})
        configuration.pop('model_type')
        configuration = RobertaConfig(**configuration)
        self.backbone = RobertaModel.from_pretrained('roberta-base', config=configuration, add_pooling_layer=False)     
        hidden_size, proj_size = 768, 256
        self.projector = nn.Linear(hidden_size, proj_size, bias=False)

    def _output_avg_pool(self, sequence_output, attention_mask):
        '''
        # This version will take padding part into calculation
        # [bs, h]
        # sequence_output_txt = F.adaptive_avg_pool1d(sequence_output_txt.transpose(1,2), 1).transpose(1,2)
        # sequence_output_img = F.adaptive_avg_pool1d(sequence_output_img.transpose(1,2), 1).transpose(1,2)
        # mask format: [1: attend / 0: ignore]
        '''
        # [bs, 1, 1]
        seq_len = attention_mask.squeeze().sum(-1, keepdim=True).unsqueeze(-1)
        # [bs, sq_len, 1]
        attention_mask = attention_mask.squeeze().unsqueeze(-1)
        # [bs, 1, h]
        pooled_output = (sequence_output * attention_mask).sum(1, keepdim=True) / seq_len
        return pooled_output.squeeze()

    def forward(self, sentence):
        latents = self.backbone(**sentence, return_dict=False)[0]
        latents = self._output_avg_pool(latents, sentence['attention_mask'])
        latents = F.linear(input=latents.float(),
                weight=self.projector.weight.float(),
            )            
        return latents

@HEADS.register_module()
class LanguageHead(BaseHead):
    """Classification head build from language encoder.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_dist_test (bool): whether distributed calculate the classname -> classweight
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 pretrained=None,
                 dataset='k400',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.pretrained = pretrained
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.language_model = LModel()
        self.visual_proj = nn.Linear(in_channels, 256, bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / 0.05)), requires_grad=False)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # ref to: https://github.com/sallymmx/ActionCLIP/blob/master/utils/Text_Prompt.py
        # self.templates = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
        #         f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
        #         f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
        #         f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
        #         f"The man is {{}}", f"The woman is {{}}"]
        self.templates = ['a photo of {}.',
            'a photo of a person {}.',
            'a photo of a person using {}.',
            'a photo of a person doing {}.',
            'a photo of a person during {}.',
            'a photo of a person performing {}.',
            'a photo of a person practicing {}.',
            'a video of {}.',
            'a video of a person {}.',
            'a video of a person using {}.',
            'a video of a person doing {}.',
            'a video of a person during {}.',
            'a video of a person performing {}.',
            'a video of a person practicing {}.',
            'a example of {}.',
            'a example of a person {}.',
            'a example of a person using {}.',
            'a example of a person doing {}.',
            'a example of a person during {}.',
            'a example of a person performing {}.',
            'a example of a person practicing {}.',
            'a demonstration of {}.',
            'a demonstration of a person {}.',
            'a demonstration of a person using {}.',
            'a demonstration of a person doing {}.',
            'a demonstration of a person during {}.',
            'a demonstration of a person performing {}.',
            'a demonstration of a person practicing {}.',
        ]
        # self.templates = [  'a bad photo of a {}.',  'a photo of many {}.',  'a sculpture of a {}.',  'a photo of the hard to see {}.',  'a low resolution photo of the {}.',  'a rendering of a {}.',  'graffiti of a {}.',  'a bad photo of the {}.',  'a cropped photo of the {}.',  'a tattoo of a {}.',  'the embroidered {}.',  'a photo of a hard to see {}.',  'a bright photo of a {}.',  'a photo of a clean {}.',  'a photo of a dirty {}.',  'a dark photo of the {}.',  'a drawing of a {}.',  'a photo of my {}.',  'the plastic {}.',  'a photo of the cool {}.',  'a close-up photo of a {}.',  'a black and white photo of the {}.',  'a painting of the {}.',  'a painting of a {}.',  'a pixelated photo of the {}.',  'a sculpture of the {}.',  'a bright photo of the {}.',  'a cropped photo of a {}.',  'a plastic {}.',  'a photo of the dirty {}.',  'a jpeg corrupted photo of a {}.',  'a blurry photo of the {}.',  'a photo of the {}.',  'a good photo of the {}.',  'a rendering of the {}.',  'a {} in a video game.',  'a photo of one {}.',  'a doodle of a {}.',  'a close-up photo of the {}.',  'a photo of a {}.',  'the origami {}.',  'the {} in a video game.',  'a sketch of a {}.',  'a doodle of the {}.',  'a origami {}.',  'a low resolution photo of a {}.',  'the toy {}.',  'a rendition of the {}.',  'a photo of the clean {}.',  'a photo of a large {}.',  'a rendition of a {}.',  'a photo of a nice {}.',  'a photo of a weird {}.',  'a blurry photo of a {}.',  'a cartoon {}.',  'art of a {}.',  'a sketch of the {}.',  'a embroidered {}.',  'a pixelated photo of a {}.',  'itap of the {}.',  'a jpeg corrupted photo of the {}.',  'a good photo of a {}.',  'a plushie {}.',  'a photo of the nice {}.',  'a photo of the small {}.',  'a photo of the weird {}.',  'the cartoon {}.',  'art of the {}.',  'a drawing of the {}.',  'a photo of the large {}.',  'a black and white photo of a {}.',  'the plushie {}.',  'a dark photo of a {}.',  'itap of a {}.',  'graffiti of the {}.',  'a toy {}.',  'itap of my {}.',  'a photo of a cool {}.',  'a photo of a small {}.',  'a tattoo of the {}.']
        # https://gist.github.com/willprice/f19da185c9c5f32847134b87c1960769
        if dataset == 'k400':
            self.classnames = ["abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel", "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)", "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing cards", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ", "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting", "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving", "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "strumming guitar", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi", "taking a shower", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", "tossing coin", "tossing salad", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck", "using computer", "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"]
        elif dataset == 'k600':
            # maybe an ill-formed evaluation protocal:
            # A few video clips from 30 classes of the Kinetics-400 validation set became part of the Kinetics-600 test set, and some from the training set became part of the new validation set. It is therefore not ideal to evaluate models on Kinetics-600 that were pre-trained on Kinetics-400
            self.classnames = ["abseiling", "acting in play", "adjusting glasses", "air drumming", "alligator wrestling", "answering questions", "applauding", "applying cream", "archaeological excavation", "archery", "arguing", "arm wrestling", "arranging flowers", "assembling bicycle", "assembling computer", "attending conference", "auctioning", "backflip (human)", "baking cookies", "bandaging", "barbequing", "bartending", "base jumping", "bathing dog", "battle rope training", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowdrying hair", "blowing bubble gum", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bodysurfing", "bookbinding", "bottling", "bouncing on bouncy castle", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "breaking boards", "breathing fire", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building lego", "building sandcastle", "building shed", "bull fighting", "bulldozing", "bungee jumping", "burping", "busking", "calculating", "calligraphy", "canoeing or kayaking", "capoeira", "capsizing", "card stacking", "card throwing", "carrying baby", "cartwheeling", "carving ice", "carving pumpkin", "casting fishing line", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing gear in car", "changing oil", "changing wheel (not on bike)", "checking tires", "cheerleading", "chewing gum", "chiseling stone", "chiseling wood", "chopping meat", "chopping vegetables", "chopping wood", "clam digging", "clapping", "clay pottery making", "clean and jerk", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "coloring in", "combing hair", "contact juggling", "contorting", "cooking egg", "cooking on campfire", "cooking sausages (not on barbeque)", "cooking scallops", "cosplaying", "counting money", "country line dancing", "cracking back", "cracking knuckles", "cracking neck", "crawling baby", "crossing eyes", "crossing river", "crying", "cumbia", "curling (sport)", "curling hair", "cutting apple", "cutting nails", "cutting orange", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "delivering mail", "dining", "directing traffic", "disc golfing", "diving cliff", "docking boat", "dodgeball", "doing aerobics", "doing jigsaw puzzle", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking shots", "driving car", "driving tractor", "drooling", "drop kicking", "drumming fingers", "dumpster diving", "dunking basketball", "dyeing eyebrows", "dyeing hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "embroidering", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "falling off bike", "falling off chair", "feeding birds", "feeding fish", "feeding goats", "fencing (sport)", "fidgeting", "finger snapping", "fixing bicycle", "fixing hair", "flint knapping", "flipping pancake", "fly tying", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "geocaching", "getting a haircut", "getting a piercing", "getting a tattoo", "giving or receiving award", "gold panning", "golf chipping", "golf driving", "golf putting", "gospel singing in church", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "hand washing clothes", "head stand", "headbanging", "headbutting", "high jump", "high kick", "historical reenactment", "hitting baseball", "hockey stop", "holding snake", "home roasting coffee", "hopscotch", "hoverboarding", "huddling", "hugging (not baby)", "hugging baby", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ice swimming", "inflating balloons", "installing carpet", "ironing", "ironing hair", "javelin throw", "jaywalking", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping bicycle", "jumping into pool", "jumping jacks", "jumpstyle dancing", "karaoke", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "land sailing", "laughing", "lawn mower racing", "laying bricks", "laying concrete", "laying stone", "laying tiles", "leatherworking", "licking", "lifting hat", "lighting fire", "lock picking", "long jump", "longboarding", "looking at phone", "luge", "lunge", "making a cake", "making a sandwich", "making balloon shapes", "making bubbles", "making cheese", "making horseshoes", "making jewelry", "making paper aeroplanes", "making pizza", "making snowman", "making sushi", "making tea", "making the bed", "marching", "marriage proposal", "massaging back", "massaging feet", "massaging legs", "massaging neck", "massaging person\'s head", "milking cow", "moon walking", "mopping floor", "mosh pit dancing", "motorcycling", "mountain climber (exercise)", "moving furniture", "mowing lawn", "mushroom foraging", "needle felting", "news anchoring", "opening bottle (not wine)", "opening door", "opening present", "opening refrigerator", "opening wine bottle", "packing", "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing american football (not in game)", "passing soccer ball", "peeling apples", "peeling potatoes", "person collecting garbage", "petting animal (not cat)", "petting cat", "photobombing", "photocopying", "picking fruit", "pillow fight", "pinching", "pirouetting", "planing wood", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing beer pong", "playing blackjack", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing darts", "playing didgeridoo", "playing dominoes", "playing drums", "playing field hockey", "playing flute", "playing gong", "playing guitar", "playing hand clapping games", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing laser tag", "playing lute", "playing maracas", "playing marbles", "playing monopoly", "playing netball", "playing ocarina", "playing organ", "playing paintball", "playing pan pipes", "playing piano", "playing pinball", "playing ping pong", "playing poker", "playing polo", "playing recorder", "playing rubiks cube", "playing saxophone", "playing scrabble", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing with trains", "playing xylophone", "poking bellybutton", "pole vault", "polishing metal", "popping balloons", "pouring beer", "preparing salad", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelbarrow", "pushing wheelchair", "putting in contact lenses", "putting on eyeliner", "putting on foundation", "putting on lipstick", "putting on mascara", "putting on sari", "putting on shoes", "raising eyebrows", "reading book", "reading newspaper", "recording music", "repairing puncture", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mule", "riding or walking with horse", "riding scooter", "riding snow blower", "riding unicycle", "ripping paper", "roasting marshmallows", "roasting pig", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "rolling pastry", "rope pushdown", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "sausage making", "sawing wood", "scrambling eggs", "scrapbooking", "scrubbing face", "scuba diving", "separating eggs", "setting table", "sewing", "shaking hands", "shaking head", "shaping bread dough", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining flashlight", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shopping", "shot put", "shoveling snow", "shucking oysters", "shuffling cards", "shuffling feet", "side kick", "sign language interpreting", "singing", "sipping cup", "situp", "skateboarding", "ski jumping", "skiing crosscountry", "skiing mono", "skiing slalom", "skipping rope", "skipping stone", "skydiving", "slacklining", "slapping", "sled dog racing", "sleeping", "smashing", "smelling feet", "smoking", "smoking hookah", "smoking pipe", "snatch weight lifting", "sneezing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spelunking", "spinning poi", "spray painting", "springboard diving", "square dancing", "squat", "standing on hands", "staring", "steer roping", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "sucking lolly", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swimming front crawl", "swing dancing", "swinging baseball bat", "swinging on something", "sword fighting", "sword swallowing", "tackling", "tagging graffiti", "tai chi", "talking on cell phone", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "tasting wine", "testifying", "texting", "threading needle", "throwing axe", "throwing ball (not baseball or American football)", "throwing discus", "throwing knife", "throwing snowballs", "throwing tantrum", "throwing water balloon", "tickling", "tie dying", "tightrope walking", "tiptoeing", "tobogganing", "tossing coin", "training dog", "trapezing", "trimming or shaving beard", "trimming shrubs", "trimming trees", "triple jump", "twiddling fingers", "tying bow tie", "tying knot (not on a tie)", "tying necktie", "tying shoe laces", "unboxing", "unloading truck", "using a microscope", "using a paint roller", "using a power drill", "using a sledge hammer", "using a wrench", "using atm", "using bagging machine", "using circular saw", "using inhaler", "using puppets", "using remote controller (not gaming)", "using segway", "vacuuming floor", "visiting the zoo", "wading through mud", "wading through water", "waiting in line", "waking up", "walking the dog", "walking through snow", "washing dishes", "washing feet", "washing hair", "washing hands", "watching tv", "water skiing", "water sliding", "watering plants", "waving hand", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "weaving fabric", "welding", "whistling", "windsurfing", "winking", "wood burning (art)", "wrapping present", "wrestling", "writing", "yarn spinning", "yawning", "yoga", "zumba"]
        elif dataset == 'k232':
            # Kinetics-600 is an approximate superset of Kinetics-400 – overall, 368 of the original 400 classes are exactly the same in Kinetics-600 (except they have more examples). 
            # For the other 32 classes, we renamed a few (e.g. “dying hair” became “dyeing hair”), split or removed others that were too strongly overlapping with other classes, such as “drinking”. 
            # We split some classes: “hugging” became “hugging baby” and “hugging (not baby)”, while “opening bottle” became “opening wine bottle” and “opening bottle (not wine)”
            # So:: this 232 classes, include 200 totally not in k400; and 32 classes which may in 400; or shifted in 400; or not in 400
            # the 32 of 400 not appear in k600: "baby waking up", "balloon blowing", "changing wheel", "cleaning floor", "cooking chicken", "cooking sausages", "digging", "drinking", "drinking beer", "dying hair", "exercising arm", "filling eyebrows", "garbage collecting", "gargling", "hugging", "making bed", "opening bottle", "passing American football (not in game)", "playing cards", "riding mountain bike", "shredding paper", "skiing (not slalom or crosscountry)", "sniffing", "spraying", "strumming guitar", "swinging legs", "taking a shower", "throwing ball", "tossing salad", "tying tie", "using computer", "vault"
            # “宝宝醒来”、“吹气球”、“换轮子”、“清洁地板”、“煮鸡肉”、“煮香肠”、“挖”、“喝”、“喝啤酒”、“染发”、“锻炼手臂”、“填眉毛”、“收集垃圾”、“漱口”、“拥抱”、“整理床铺”、“打开瓶子”、“传球美式足球（非比赛中）”、“扑克牌”，“骑山地车”、“碎纸”、“滑雪（非回转或越野）”、“嗅探”、“喷洒”、“弹吉他”、“摆动腿”、“洗澡”、“扔球”、“扔沙拉”、“系领带”、“使用电脑”、“跳马”
            # this is the 232 total appear in 600 not appear in 400;
            self.classnames = ["acting in play", "adjusting glasses", "alligator wrestling", "archaeological excavation", "arguing", "assembling bicycle", "attending conference", "backflip (human)", "base jumping", "bathing dog", "battle rope training", "blowdrying hair", "blowing bubble gum", "bodysurfing", "bottling", "bouncing on bouncy castle", "breaking boards", "breathing fire", "building lego", "building sandcastle", "bull fighting", "bulldozing", "burping", "calculating", "calligraphy", "capsizing", "card stacking", "card throwing", "carving ice", "casting fishing line", "changing gear in car", "changing wheel (not on bike)", "chewing gum", "chiseling stone", "chiseling wood", "chopping meat", "chopping vegetables", "clam digging", "coloring in", "combing hair", "contorting", "cooking sausages (not on barbeque)", "cooking scallops", "cosplaying", "cracking back", "cracking knuckles", "crossing eyes", "cumbia", "curling (sport)", "cutting apple", "cutting orange", "delivering mail", "directing traffic", "docking boat", "doing jigsaw puzzle", "drooling", "dumpster diving", "dyeing eyebrows", "dyeing hair", "embroidering", "falling off bike", "falling off chair", "fencing (sport)", "fidgeting", "fixing bicycle", "flint knapping", "fly tying", "geocaching", "getting a piercing", "gold panning", "gospel singing in church", "hand washing clothes", "head stand", "historical reenactment", "home roasting coffee", "huddling", "hugging (not baby)", "hugging baby", "ice swimming", "inflating balloons", "installing carpet", "ironing hair", "jaywalking", "jumping bicycle", "jumping jacks", "karaoke", "land sailing", "lawn mower racing", "laying concrete", "laying stone", "laying tiles", "leatherworking", "licking", "lifting hat", "lighting fire", "lock picking", "longboarding", "looking at phone", "luge", "making balloon shapes", "making bubbles", "making cheese", "making horseshoes", "making paper aeroplanes", "making the bed", "marriage proposal", "massaging neck", "moon walking", "mosh pit dancing", "mountain climber (exercise)", "mushroom foraging", "needle felting", "opening bottle (not wine)", "opening door", "opening refrigerator", "opening wine bottle", "packing", "passing american football (not in game)", "passing soccer ball", "person collecting garbage", "photobombing", "photocopying", "pillow fight", "pinching", "pirouetting", "planing wood", "playing beer pong", "playing blackjack", "playing darts", "playing dominoes", "playing field hockey", "playing gong", "playing hand clapping games", "playing laser tag", "playing lute", "playing maracas", "playing marbles", "playing netball", "playing ocarina", "playing pan pipes", "playing pinball", "playing ping pong", "playing polo", "playing rubiks cube", "playing scrabble", "playing with trains", "poking bellybutton", "polishing metal", "popping balloons", "pouring beer", "preparing salad", "pushing wheelbarrow", "putting in contact lenses", "putting on eyeliner", "putting on foundation", "putting on lipstick", "putting on mascara", "putting on sari", "putting on shoes", "raising eyebrows", "repairing puncture", "riding snow blower", "roasting marshmallows", "roasting pig", "rolling pastry", "rope pushdown", "sausage making", "sawing wood", "scrapbooking", "scrubbing face", "separating eggs", "sewing", "shaping bread dough", "shining flashlight", "shopping", "shucking oysters", "shuffling feet", "sipping cup", "skiing mono", "skipping stone", "sleeping", "smashing", "smelling feet", "smoking pipe", "spelunking", "square dancing", "standing on hands", "staring", "steer roping", "sucking lolly", "swimming front crawl", "swinging baseball bat", "sword swallowing", "tackling", "tagging graffiti", "talking on cell phone", "tasting wine", "threading needle", "throwing ball (not baseball or American football)", "throwing knife", "throwing snowballs", "throwing tantrum", "throwing water balloon", "tie dying", "tightrope walking", "tiptoeing", "trimming shrubs", "twiddling fingers", "tying necktie", "tying shoe laces", "using a microscope", "using a paint roller", "using a power drill", "using a sledge hammer", "using a wrench", "using atm", "using bagging machine", "using circular saw", "using inhaler", "using puppets", "vacuuming floor", "visiting the zoo", "wading through mud", "wading through water", "waking up", "walking through snow", "watching tv", "waving hand", "weaving fabric", "winking", "wood burning (art)", "yarn spinning"]

            # the 32 from k400 to map to the one in 232(k600)
            # baby waking up -> waking up
            # balloon blowing -> inflating balloons
            # changing wheel -> changing wheel (not on bike)
            # cleaning floor -> vacuuming floor
            # cooking chicken -> x
            # cooking sausages -> cooking sausages (not on barbeque)
            # digging -> clam digging ??
            # drinking -> x
            # drinking beer -> x
            # dying hair -> dyeing hair
            # exercising arm -> x
            # filling eyebrows -> dyeing eyebrows
            # garbage collecting -> person collecting garbage
            # gargling -> ??
            # hugging -> hugging baby ; hugging (not baby)
            # making bed -> making the bed
            # opening bottle -> opening wine bottle ; opening bottle (not wine)
            # passing American football (not in game) -> passing american football (not in game)
            # playing cards -> card stacking ; card throwing
            # riding mountain bike -> x
            # shredding paper -> x
            # skiing (not slalom or crosscountry) -> skiing mono
            # sniffing -> ?
            # spraying -> ?
            # strumming guitar -> x
            # swinging legs -> x
            # taking a shower -> x
            # throwing ball -> throwing ball (not baseball or American football)
            # tossing salad -> preparing salad
            # tying tie -> tying necktie
            # using computer -> x
            # vault -> x
        elif dataset == 'k368':
            # k600 -- k232
            self.classnames = ["abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baking cookies", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking egg", "cooking on campfire", "counting money", "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking shots", "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging person\'s head", "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring", "opening present", "paragliding", "parasailing", "parkour", "passing American football (in game)", "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ", "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow", "shuffling cards", "side kick", "sign language interpreting", "singing", "situp", "skateboarding", "ski jumping", "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spinning poi", "spray painting", "springboard diving", "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging on something", "sword fighting", "tai chi", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing discus", "tickling", "tobogganing", "tossing coin", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)", "unboxing", "unloading truck", "using remote controller (not gaming)", "using segway", "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"]
        elif dataset == 'k215':
            # mannually filter the 232 classes and see if it in 32 classes.
            self.classnames = ["acting in play", "adjusting glasses", "alligator wrestling", "archaeological excavation", "arguing", "assembling bicycle", "attending conference", "backflip (human)", "base jumping", "bathing dog", "battle rope training", "blowdrying hair", "blowing bubble gum", "bodysurfing", "bottling", "bouncing on bouncy castle", "breaking boards", "breathing fire", "building lego", "building sandcastle", "bull fighting", "bulldozing", "burping", "calculating", "calligraphy", "capsizing", "carving ice", "casting fishing line", "changing gear in car", "chewing gum", "chiseling stone", "chiseling wood", "chopping meat", "chopping vegetables", "clam digging", "coloring in", "combing hair", "contorting", "cooking scallops", "cosplaying", "cracking back", "cracking knuckles", "crossing eyes", "cumbia", "curling (sport)", "cutting apple", "cutting orange", "delivering mail", "directing traffic", "docking boat", "doing jigsaw puzzle", "drooling", "dumpster diving", "embroidering", "falling off bike", "falling off chair", "fencing (sport)", "fidgeting", "fixing bicycle", "flint knapping", "fly tying", "geocaching", "getting a piercing", "gold panning", "gospel singing in church", "hand washing clothes", "head stand", "historical reenactment", "home roasting coffee", "huddling", "ice swimming", "installing carpet", "ironing hair", "jaywalking", "jumping bicycle", "jumping jacks", "karaoke", "land sailing", "lawn mower racing", "laying concrete", "laying stone", "laying tiles", "leatherworking", "licking", "lifting hat", "lighting fire", "lock picking", "longboarding", "looking at phone", "luge", "making balloon shapes", "making bubbles", "making cheese", "making horseshoes", "making paper aeroplanes", "making the bed", "marriage proposal", "massaging neck", "moon walking", "mosh pit dancing", "mountain climber (exercise)", "mushroom foraging", "needle felting", "opening door", "opening refrigerator", "packing", "passing soccer ball", "person collecting garbage", "photobombing", "photocopying", "pillow fight", "pinching", "pirouetting", "planing wood", "playing beer pong", "playing blackjack", "playing darts", "playing dominoes", "playing field hockey", "playing gong", "playing hand clapping games", "playing laser tag", "playing lute", "playing maracas", "playing marbles", "playing netball", "playing ocarina", "playing pan pipes", "playing pinball", "playing ping pong", "playing polo", "playing rubiks cube", "playing scrabble", "playing with trains", "poking bellybutton", "polishing metal", "popping balloons", "pouring beer", "pushing wheelbarrow", "putting in contact lenses", "putting on eyeliner", "putting on foundation", "putting on lipstick", "putting on mascara", "putting on sari", "putting on shoes", "raising eyebrows", "repairing puncture", "riding snow blower", "roasting marshmallows", "roasting pig", "rolling pastry", "rope pushdown", "sausage making", "sawing wood", "scrapbooking", "scrubbing face", "separating eggs", "sewing", "shaping bread dough", "shining flashlight", "shopping", "shucking oysters", "shuffling feet", "sipping cup", "skipping stone", "sleeping", "smashing", "smelling feet", "smoking pipe", "spelunking", "square dancing", "standing on hands", "staring", "steer roping", "sucking lolly", "swimming front crawl", "swinging baseball bat", "sword swallowing", "tackling", "tagging graffiti", "talking on cell phone", "tasting wine", "threading needle", "throwing knife", "throwing snowballs", "throwing tantrum", "throwing water balloon", "tie dying", "tightrope walking", "tiptoeing", "trimming shrubs", "twiddling fingers", "tying shoe laces", "using a microscope", "using a paint roller", "using a power drill", "using a sledge hammer", "using a wrench", "using atm", "using bagging machine", "using circular saw", "using inhaler", "using puppets", "visiting the zoo", "wading through mud", "wading through water", "waking up", "walking through snow", "watching tv", "waving hand", "weaving fabric", "winking", "wood burning (art)", "yarn spinning"]
        elif dataset == 'sthv2':
            self.classnames = ['Approaching something with your camera', 'Attaching something to something', 'Bending something so that it deforms', 'Bending something until it breaks', 'Burying something in something', 'Closing something', 'Covering something with something', 'Digging something out of something', 'Dropping something behind something', 'Dropping something in front of something', 'Dropping something into something', 'Dropping something next to something', 'Dropping something onto something', 'Failing to put something into something because something does not fit', 'Folding something', 'Hitting something with something', 'Holding something', 'Holding something behind something', 'Holding something in front of something', 'Holding something next to something', 'Holding something over something', 'Laying something on the table on its side, not upright', 'Letting something roll along a flat surface', 'Letting something roll down a slanted surface', 'Letting something roll up a slanted surface, so it rolls back down', 'Lifting a surface with something on it but not enough for it to slide down', 'Lifting a surface with something on it until it starts sliding down', 'Lifting something up completely without letting it drop down', 'Lifting something up completely, then letting it drop down', 'Lifting something with something on it', 'Lifting up one end of something without letting it drop down', 'Lifting up one end of something, then letting it drop down', 'Moving away from something with your camera', 'Moving part of something', 'Moving something across a surface until it falls down', 'Moving something across a surface without it falling down', 'Moving something and something away from each other', 'Moving something and something closer to each other', 'Moving something and something so they collide with each other', 'Moving something and something so they pass each other', 'Moving something away from something', 'Moving something away from the camera', 'Moving something closer to something', 'Moving something down', 'Moving something towards the camera', 'Moving something up', 'Opening something', 'Picking something up', 'Piling something up', 'Plugging something into something', 'Plugging something into something but pulling it right out as you remove your hand', 'Poking a hole into some substance', 'Poking a hole into something soft', 'Poking a stack of something so the stack collapses', 'Poking a stack of something without the stack collapsing', 'Poking something so it slightly moves', "Poking something so lightly that it doesn't or almost doesn't move", 'Poking something so that it falls over', 'Poking something so that it spins around', 'Pouring something into something', 'Pouring something into something until it overflows', 'Pouring something onto something', 'Pouring something out of something', 'Pretending or failing to wipe something off of something', 'Pretending or trying and failing to twist something', 'Pretending to be tearing something that is not tearable', 'Pretending to close something without actually closing it', 'Pretending to open something without actually opening it', 'Pretending to pick something up', 'Pretending to poke something', 'Pretending to pour something out of something, but something is empty', 'Pretending to put something behind something', 'Pretending to put something into something', 'Pretending to put something next to something', 'Pretending to put something on a surface', 'Pretending to put something onto something', 'Pretending to put something underneath something', 'Pretending to scoop something up with something', 'Pretending to spread air onto something', 'Pretending to sprinkle air onto something', 'Pretending to squeeze something', 'Pretending to take something from somewhere', 'Pretending to take something out of something', 'Pretending to throw something', 'Pretending to turn something upside down', 'Pulling something from behind of something', 'Pulling something from left to right', 'Pulling something from right to left', 'Pulling something onto something', 'Pulling something out of something', 'Pulling two ends of something but nothing happens', 'Pulling two ends of something so that it gets stretched', 'Pulling two ends of something so that it separates into two pieces', 'Pushing something from left to right', 'Pushing something from right to left', 'Pushing something off of something', 'Pushing something onto something', 'Pushing something so it spins', "Pushing something so that it almost falls off but doesn't", 'Pushing something so that it falls off the table', 'Pushing something so that it slightly moves', 'Pushing something with something', 'Putting number of something onto something', 'Putting something and something on the table', 'Putting something behind something', 'Putting something in front of something', 'Putting something into something', 'Putting something next to something', 'Putting something on a flat surface without letting it roll', 'Putting something on a surface', 'Putting something on the edge of something so it is not supported and falls down', "Putting something onto a slanted surface but it doesn't glide down", 'Putting something onto something', 'Putting something onto something else that cannot support it so it falls down', 'Putting something similar to other things that are already on the table', "Putting something that can't roll onto a slanted surface, so it slides down", "Putting something that can't roll onto a slanted surface, so it stays where it is", 'Putting something that cannot actually stand upright upright on the table, so it falls on its side', 'Putting something underneath something', 'Putting something upright on the table', 'Putting something, something and something on the table', 'Removing something, revealing something behind', 'Rolling something on a flat surface', 'Scooping something up with something', 'Showing a photo of something to the camera', 'Showing something behind something', 'Showing something next to something', 'Showing something on top of something', 'Showing something to the camera', 'Showing that something is empty', 'Showing that something is inside something', 'Something being deflected from something', 'Something colliding with something and both are being deflected', 'Something colliding with something and both come to a halt', 'Something falling like a feather or paper', 'Something falling like a rock', 'Spilling something behind something', 'Spilling something next to something', 'Spilling something onto something', 'Spinning something so it continues spinning', 'Spinning something that quickly stops spinning', 'Spreading something onto something', 'Sprinkling something onto something', 'Squeezing something', 'Stacking number of something', 'Stuffing something into something', 'Taking one of many similar things on the table', 'Taking something from somewhere', 'Taking something out of something', 'Tearing something into two pieces', 'Tearing something just a little bit', 'Throwing something', 'Throwing something against something', 'Throwing something in the air and catching it', 'Throwing something in the air and letting it fall', 'Throwing something onto a surface', "Tilting something with something on it slightly so it doesn't fall down", 'Tilting something with something on it until it falls off', 'Tipping something over', 'Tipping something with something in it over, so something in it falls out', 'Touching (without moving) part of something', "Trying but failing to attach something to something because it doesn't stick", 'Trying to bend something unbendable so nothing happens', 'Trying to pour something into something, but missing so it spills next to it', 'Turning something upside down', 'Turning the camera downwards while filming something', 'Turning the camera left while filming something', 'Turning the camera right while filming something', 'Turning the camera upwards while filming something', 'Twisting (wringing) something wet until water comes out', 'Twisting something', 'Uncovering something', 'Unfolding something', 'Wiping something off of something']
        elif dataset == 'ucf101':
            self.classnames = [    
            'Apply Eye Makeup',
            'Apply Lipstick',
            'Archery',
            'Baby Crawling',
            'Balance Beam',
            'Band Marching',
            'Baseball Pitch',
            'Basketball',
            'Basketball Dunk',
            'Bench Press',
            'Biking',
            'Billiards',
            'Blow Dry Hair',
            'Blowing Candles',
            'Body Weight Squats',
            'Bowling',
            'Boxing Punching Bag',
            'Boxing Speed Bag',
            'Breast Stroke',
            'Brushing Teeth',
            'Clean And Jerk',
            'Cliff Diving',
            'Cricket Bowling',
            'Cricket Shot',
            'Cutting In Kitchen',
            'Diving',
            'Drumming',
            'Fencing',
            'Field Hockey Penalty',
            'Floor Gymnastics',
            'Frisbee Catch',
            'Front Crawl',
            'Golf Swing',
            'Haircut',
            'Hammer Throw',
            'Hammering',
            'Hand Stand Pushups',
            'Handstand Walking',
            'Head Massage',
            'High Jump',
            'Horse Race',
            'Horse Riding',
            'Hula Hoop',
            'Ice Dancing',
            'Javelin Throw',
            'Juggling Balls',
            'Jump Rope',
            'Jumping Jack',
            'Kayaking',
            'Knitting',
            'Long Jump',
            'Lunges',
            'Military Parade',
            'Mixing',
            'Mopping Floor',
            'Nunchucks',
            'Parallel Bars',
            'Pizza Tossing',
            'Playing Cello',
            'Playing Daf',
            'Playing Dhol',
            'Playing Flute',
            'Playing Guitar',
            'Playing Piano',
            'Playing Sitar',
            'Playing Tabla',
            'Playing Violin',
            'Pole Vault',
            'Pommel Horse',
            'Pull Ups',
            'Punch',
            'Push Ups',
            'Rafting',
            'Rock Climbing Indoor',
            'Rope Climbing',
            'Rowing',
            'Salsa Spin',
            'Shaving Beard',
            'Shotput',
            'Skate Boarding',
            'Skiing',
            'Skijet',
            'Sky Diving',
            'Soccer Juggling',
            'Soccer Penalty',
            'Still Rings',
            'Sumo Wrestling',
            'Surfing',
            'Swing',
            'Table Tennis Shot',
            'Tai Chi',
            'Tennis Swing',
            'Throw Discus',
            'Trampoline Jumping',
            'Typing',
            'Uneven Bars',
            'Volleyball Spiking',
            'Walking With Dog',
            'Wall Pushups',
            'Writing On Board',
            'Yo Yo',
        ]
        elif dataset == 'hmdb51':
            self.classnames = [
                'brush hair',
                'cartwheel',
                'catch',
                'chew',
                'clap',
                'clap',
                'climb',
                'climb stairs',
                'dive',
                'draw sword',
                'dribble',
                'drink',
                'eat',
                'fall floor',
                'fencing',
                'flic flac',
                'golf',
                'handstand',
                'hit',
                'hug',
                'jump',
                'kick',
                'kick ball',
                'kiss',
                'laugh',
                'pick',
                'pour',
                'pullup',
                'punch',
                'push',
                'pushup',
                'ride bike',
                'ride horse',
                'run',
                'shake hands',
                'shoot ball',
                'shoot bow',
                'shoot gun',
                'sit',
                'situp',
                'smile',
                'smoke',
                'somersault',
                'stand',
                'swing baseball',
                'sword',
                'sword exercise',
                'talk',
                'throw',
                'turn',
                'walk',
                'wave']
        global_rank = int(dist.get_rank())
        total_gpu = int(dist.get_world_size())
        # if dataset != 'sthv2' and dataset != 'ucf101':
        #     assert len(self.classnames) % total_gpu == 0, " to distributed class name, should seperatable"
        #     per_gpu_len = len(self.classnames) // total_gpu
        # else:
        per_gpu_len = len(self.classnames) // total_gpu + 1
        self.full_classnames = self.classnames
        self.classnames = self.classnames[global_rank * per_gpu_len : min((global_rank+1) * per_gpu_len, len(self.classnames))]

        self.class_head_weight_gathered_promptensembled = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        ### LOAD Pretrain Weight
        logger = get_root_logger()
        if self.pretrained is not None:
            assert isinstance(self.pretrained, str), 'give path to pretrained vl model'
            logger.info(f'load language head from: {self.pretrained}')
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']

            new_state_dict = {}
            new_state_dict_proj = {}
            for k, v in state_dict.items():
                if k.startswith('sentence_model.'):
                    new_state_dict[k.replace('sentence_model.', '')] = v.cpu()
                if k.startswith('visual_model.projector.'):
                    new_state_dict_proj[k.replace('visual_model.projector.', '')] = v.cpu()
            state_dict = new_state_dict
            msg = self.language_model.load_state_dict(state_dict, strict=False)
            self.visual_proj.load_state_dict(new_state_dict_proj, strict=True)

            logger.info(msg)
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            logger.warning(" You have to use pretrained language model for now!! If it's for test, skip it")

    def forward_clshead(self, device, ensembled=False):
        ## inference class weight:
        if not ensembled:
            sentence = []
            for sent in self.classnames:
                prompt = random.choice(self.templates)
                sentence.append(prompt.format(sent).lower())
            prompted_imagenet_classhead_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=16, return_tensors='pt')
            prompted_imagenet_classhead_input = {k:v.to(device, non_blocking=True) for k, v in prompted_imagenet_classhead_input.items()}
            # with torch.no_grad():
            class_head_weight = self.language_model(prompted_imagenet_classhead_input)
            # class_head_weight_gathered = SyncFunction.apply(class_head_weight) 
            class_head_weight_gathered = varsize_dist_collect(class_head_weight)
            return class_head_weight_gathered
        else:
            print("Begin Ensemble Forward")
            zeroshot_weights = []
            for idx, classname in enumerate(self.full_classnames):
                texts = []
                for template in self.templates:
                    _texts = template.format(classname)
                    texts.append(_texts.lower())
                prompted_imagenet_classhead_input = self.tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors='pt')
                prompted_imagenet_classhead_input = {k:v.to(device, non_blocking=True) for k, v in prompted_imagenet_classhead_input.items()}
                with torch.no_grad():
                    class_head_weight = self.language_model(prompted_imagenet_classhead_input)
                    class_head_weight /= class_head_weight.norm(dim=-1, keepdim=True)
                    class_embedding = class_head_weight.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device).t()
            return zeroshot_weights


    def forward(self, x, is_test=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        x = self.visual_proj(x)
        if not is_test:
            class_head_weight_gathered = self.forward_clshead(x.device)
        else:
            if self.class_head_weight_gathered_promptensembled is not None:
                class_head_weight_gathered = self.class_head_weight_gathered_promptensembled
            else:
                self.class_head_weight_gathered_promptensembled = self.forward_clshead(x.device, ensembled=True)
                class_head_weight_gathered = self.class_head_weight_gathered_promptensembled

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        x, class_head_weight_gathered, = \
            map(lambda t: F.normalize(t, p = 2, dim = -1) if t is not None else t, (x, class_head_weight_gathered))
        # [N, num_classes]
        cls_score = logit_scale * x @ class_head_weight_gathered.t()
        return cls_score
