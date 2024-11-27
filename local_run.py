
# env
# pip install git+https://github.com/open-mmlab/mim.git
# mim install mmaction2 ##~/.local/bin/mim

# bash tools/dist_train.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py 8 --cfg-options model.backbone.pretrained=22KLaion30EP_20211014.pth

# test k600
bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py /exp/output/2021-10-14/IN21KSUP_LaionSStockCC15MVL/k400_lanhead/epoch_30.pth 8 --eval top_k_accuracy
--cfg-options model.cls_head.dataset=k600 data.test.ann_file='data/kinetics600/kinetics600_val_list.txt' data.test.data_prefix='data/kinetics600/val'
# test k400
bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py /exp/output/2021-10-14/IN21KSUP_LaionSStockCC15MVL/k400_lanhead/epoch_30.pth 8 --eval top_k_accuracy --cfg-options model.cls_head.dataset=k400 data.test.ann_file='data/kinetics400/kinetics400_val_list.txt' data.test.data_prefix='data/kinetics400/val'
# test k232
bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py ./epoch_30.pth 16 --eval top_k_accuracy --cfg-options model.cls_head.dataset=k232 data.test.ann_file='data/kinetics600/kinetics232_val_list_in600idx.txt' data.test.data_prefix='data/kinetics600/val'
# test ucf101
bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py ./epoch_30.pth 4 --eval top_k_accuracy --cfg-options model.cls_head.dataset=ucf101 data.test.ann_file='data/ucf101/ucf101_val_split_1_videos.txt' data.test.data_prefix='data/ucf101/videos'

bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py ./epoch_30.pth 4 --eval top_k_accuracy --cfg-options model.cls_head.dataset=hmdb51 data.test.ann_file='data/hmdb51/testlist01.txt' data.test.data_prefix='data/hmdb51/videos'

# for zs: maybe need to hack set pretrain path in model.backbone and model.cls_head
# mean=[123.25239296, 117.20384, 104.50194688], std=[68.76916224, 66.89346048, 70.59894016], to_bgr=False)
# bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_lanhead.py PLACEHOLD 8 --eval top_k_accuracy --cfg-option model.cls_head.dataset=k232 data.test.ann_file='data/kinetics600/kinetics232_val_list.txt' data.test.data_prefix='data/kinetics600/val'

azcopy copy https://vlpretraineastus.blob.core.windows.net/exp/output/2021-11-07/IN21KSup_LaionSStockCC15MVL_FIXDATA_withdef_truncatelonger_resume_lrsmaller_7node/lvis_frcnn_4c_1f_bb0.1_lanhead_ALLCLASSES_tau0.03_withdef/epoch_24.pth\?st\=2021-08-05T04%3A04%3A59Z\&se\=2021-12-06T04%3A04%3A00Z\&sp\=racwdl\&sv\=2018-03-28\&sr\=c\&sig\=xavxaOzfNJt59ALG6Iix8r1%2FnHs7l9qfQgdUeBKBH6I%3D ./


bash tools/dist_test.sh configs/recognition/swin/swin_base_patch244_window81212_kinetics400_22k_lanhead_384.py /msrhyper-weka/yixuanwei/IN21KSup_LaionSStockCC15MVL_FIXDATA_withdef_truncatelonger_resume_lrsmaller_7node/k400_lanhead_384_fixloading/epoch_30.pth 16 --eval top_k_accuracy --cfg-options model.cls_head.dataset=k400 data.test.ann_file='data/kinetics400/kinetics400_val_list.txt' data.test.data_prefix='data/kinetics400/val'