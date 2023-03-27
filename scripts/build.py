# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Rao Fu, RainbowSecret
# --------------------------------------------------------

#from pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
#from .hrt import HighResolutionTransformer
#from seg_hrt import SegmentationHrtModule


# CWDE: import removed to replace it with the versions of these files in lib/models/backbones/hrnet
# from pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
from lib.models.backbones.backbone_selector import BackboneSelector

#from models.seg_hrt import SegmentationHrtModule <- not implmeneted TODO:
import wandb

# Modified by CWDE: Backbone is selected through the Config object param. model is None if this argument is invalid.
# Modified Engut, removing checks to leave only hrnet and hrt, added wandb_run param
def build_model(config, wandb_run):
    model_type_sel= config.net['BACKBONE']
    backbone_selector = BackboneSelector(config = config)
    model = backbone_selector.get_backbone(wandb_run = wandb_run)
    
    #elif model_type_sel == "hrt":
    #    mode = SegmentationHrtModule(
    #        config.MODEL.HRT, wandb_run=wandb_run
    #    )
        #model = HighResolutionTransformer(
        #    config.MODEL.HRT, num_classes=config.MODEL.NUM_CLASSES
        #)
        # Considering config.MODEL.HRT
    
    if not model:
        raise NotImplementedError(f"Unknown model: {model_type_sel}")
    
    print(model)
    return model
