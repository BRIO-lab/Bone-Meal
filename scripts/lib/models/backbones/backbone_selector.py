##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from lib.models.backbones.resnet.resnet_backbone import ResNetBackbone
#from lib.models.backbones.hrnet.hrnet_backbone import HRNetBackbone

from lib.models.backbones.hrt.hrt import *
from lib.models.backbones.hrnet.pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
#from lib.models.backbones.swin.swin_backbone import SwinTransformerBackbone
from lib.utils.tools.logger import Logger as Log

"""
Zach Curran
Inputs: Config 
Outputs: Selected backbone
Rationale: Used to select the backbone specified in the config dictionary
Future: Add more choices to account for additional backbones
"""

class BackboneSelector(object):
    def __init__(self, config):
        self.config = config

    def get_backbone(self, wandb_run = None, **params):
        
        # CWDE
        # backbone (renamed to choice) now receives the option in config.net['BACKBONE'] through a Config object.
        #backbone = self.configer.get("network", "backbone")
        choice = self.config.net['BACKBONE']

        model = None

        if choice == 'hrt_small':
            model = HRT_SMALL_OCR_V2(config = self.config)

        elif choice == 'hrnet':
            model = PoseHighResolutionNet(
                                        num_key_points=self.config.segmentation_net_module['NUM_KEY_POINTS'],
                                        num_image_channels=self.config.segmentation_net_module['NUM_IMG_CHANNELS'])
        else:
            Log.error("Backbone {} is invalid.".format(backbone))
            exit(1)

        return model
