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
from pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
#from models.seg_hrt import SegmentationHrtModule <- not implmeneted TODO:
import wandb

# Modified by Engut, removing checks to leave only hrnet and hrt, added wandb_run param
def build_model(config, wandb_run):
    model_type_sel = config.dataset['MODEL_TYPE']
    if model_type_sel == "fem":
        model = SegmentationNetModule(
            config=config, wandb_run=wandb_run
        )
        

    #elif model_type_sel == "hrt":
    #    mode = SegmentationHrtModule(
    #        config.MODEL.HRT, wandb_run=wandb_run
    #    )
        #model = HighResolutionTransformer(
        #    config.MODEL.HRT, num_classes=config.MODEL.NUM_CLASSES
        #)
        # Considering config.MODEL.HRT

    else:
        raise NotImplementedError(f"Unkown model: {model_type_sel}")

    print(model)
    return model
