import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import normal_init
from mmaction.utils import get_root_logger

from ..builder import HEADS
from .base import BaseHead

import random
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

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
        self.visual_proj = nn.Linear(768, 256, bias=False)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / 0.05)), requires_grad=False)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # ref to: https://github.com/sallymmx/ActionCLIP/blob/master/utils/Text_Prompt.py
        self.templates = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
        # self.templates = [  'a bad photo of a {}.',  'a photo of many {}.',  'a sculpture of a {}.',  'a photo of the hard to see {}.',  'a low resolution photo of the {}.',  'a rendering of a {}.',  'graffiti of a {}.',  'a bad photo of the {}.',  'a cropped photo of the {}.',  'a tattoo of a {}.',  'the embroidered {}.',  'a photo of a hard to see {}.',  'a bright photo of a {}.',  'a photo of a clean {}.',  'a photo of a dirty {}.',  'a dark photo of the {}.',  'a drawing of a {}.',  'a photo of my {}.',  'the plastic {}.',  'a photo of the cool {}.',  'a close-up photo of a {}.',  'a black and white photo of the {}.',  'a painting of the {}.',  'a painting of a {}.',  'a pixelated photo of the {}.',  'a sculpture of the {}.',  'a bright photo of the {}.',  'a cropped photo of a {}.',  'a plastic {}.',  'a photo of the dirty {}.',  'a jpeg corrupted photo of a {}.',  'a blurry photo of the {}.',  'a photo of the {}.',  'a good photo of the {}.',  'a rendering of the {}.',  'a {} in a video game.',  'a photo of one {}.',  'a doodle of a {}.',  'a close-up photo of the {}.',  'a photo of a {}.',  'the origami {}.',  'the {} in a video game.',  'a sketch of a {}.',  'a doodle of the {}.',  'a origami {}.',  'a low resolution photo of a {}.',  'the toy {}.',  'a rendition of the {}.',  'a photo of the clean {}.',  'a photo of a large {}.',  'a rendition of a {}.',  'a photo of a nice {}.',  'a photo of a weird {}.',  'a blurry photo of a {}.',  'a cartoon {}.',  'art of a {}.',  'a sketch of the {}.',  'a embroidered {}.',  'a pixelated photo of a {}.',  'itap of the {}.',  'a jpeg corrupted photo of the {}.',  'a good photo of a {}.',  'a plushie {}.',  'a photo of the nice {}.',  'a photo of the small {}.',  'a photo of the weird {}.',  'the cartoon {}.',  'art of the {}.',  'a drawing of the {}.',  'a photo of the large {}.',  'a black and white photo of a {}.',  'the plushie {}.',  'a dark photo of a {}.',  'itap of a {}.',  'graffiti of the {}.',  'a toy {}.',  'itap of my {}.',  'a photo of a cool {}.',  'a photo of a small {}.',  'a tattoo of the {}.']
        # https://gist.github.com/willprice/f19da185c9c5f32847134b87c1960769
        self.classnames = ["abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel", "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)", "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing cards", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ", "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting", "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving", "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "strumming guitar", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi", "taking a shower", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", "tossing coin", "tossing salad", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck", "using computer", "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"]
        global_rank = int(dist.get_rank())
        total_gpu = int(dist.get_world_size())
        assert len(self.classnames) % total_gpu == 0, " to distributed class name, should seperatable"
        per_gpu_len = len(self.classnames) // total_gpu
        self.classnames = self.classnames[global_rank * per_gpu_len : min((global_rank+1) * per_gpu_len, len(self.classnames))]

    def init_weights(self, pretrained=None):
        """Initiate the parameters from scratch."""
        ### LOAD Pretrain Weight
        if pretrained:
            self.pretrained = pretrained
        assert isinstance(self.pretrained, str), 'give path to pretrained vl model'
        logger = get_root_logger()
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

    def forward(self, x):
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
        ## inference class weight:
        sentence = []
        for sent in self.classnames:
            prompt = random.choice(self.templates)
            sentence.append(prompt.format(sent))
        prompted_imagenet_classhead_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=16, return_tensors='pt')
        prompted_imagenet_classhead_input = {k:v.to(x.device, non_blocking=True) for k, v in prompted_imagenet_classhead_input.items()}
        # with torch.no_grad():
        class_head_weight = self.language_model(prompted_imagenet_classhead_input)
        class_head_weight_gathered = SyncFunction.apply(class_head_weight) 
        # class_head_weight_gathered = varsize_dist_collect(class_head_weight)
        
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        x, class_head_weight_gathered, = \
            map(lambda t: F.normalize(t, p = 2, dim = -1) if t is not None else t, (x, class_head_weight_gathered))
        # [N, num_classes]
        cls_score = logit_scale * x @ class_head_weight_gathered.t()
        return cls_score
