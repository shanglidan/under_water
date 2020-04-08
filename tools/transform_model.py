import torch
import math
num_classes =2
num_anchor = 5
model_coco = torch.load("./data/pretrain/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth")
    
# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][:num_classes, :]
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][:num_classes, :]
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][:num_classes, :]
# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
#anchor
model_coco["state_dict"]["rpn_head.rpn_cls.weight"] = model_coco["state_dict"]["rpn_head.rpn_cls.weight"].repeat(math.ceil(num_anchor/3),1, 1, 1)[:num_anchor,:,:,:]
model_coco["state_dict"]["rpn_head.rpn_cls.bias"] = model_coco["state_dict"]["rpn_head.rpn_cls.bias"].repeat(math.ceil(num_anchor/3))[:num_anchor]
model_coco["state_dict"]["rpn_head.rpn_reg.weight"] = model_coco["state_dict"]["rpn_head.rpn_reg.weight"].repeat(math.ceil(num_anchor/3),1,1,1)[:num_anchor*4,:,:,:]
model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = model_coco["state_dict"]["rpn_head.rpn_reg.bias"].repeat(math.ceil(num_anchor/3))[:num_anchor*4]
#save new model
torch.save(model_coco,"./data/pretrain/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc_c%d_a%d.pth"%(num_classes,num_anchor))

