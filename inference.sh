CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/front/20200404_cascade_rcnn_dconv_c3-c5_r101_fpn_1x_lr00125_600_1000_vflip.py work_dirs_front/20200331_cas_dconv_r101_fpn_1x_lr00125_600_1000_vflip/epoch_12.pth --json_out data/results/front_r101.json
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/side/20200404_cascade_rcnn_dconv_c3-c5_r101_fpn_1x_lr005_600_800_Vflip_vflip.py work_dirs_side/20200404_cas_dconv_r101_fpn_1x_lr005_600_800_Vflip/epoch_12.pth --json_out data/results/side_r101_1.json

CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/side/20200404_cascade_rcnn_dconv_c3-c5_se101_fpn_1x_lr005_600_800_Vflip_vflip.py work_dirs_side/20200404_cas_dconv_se101_fpn_1x_lr005_600_800_Vflip/epoch_12.pth --json_out data/results/side_se101.json

CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/side/20200331_cascade_rcnn_dconv_c3-c5_r101_fpn_1x_lr005_600_800_Vflip_vflip.py work_dirs_side/20200331_cas_dconv_r101_fpn_1x_lr005_600_800_Vflip/epoch_12.pth --json_out data/results/side_r101_2.json