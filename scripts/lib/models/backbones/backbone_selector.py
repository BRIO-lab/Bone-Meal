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
from lib.models.backbones.hrt.hrt_backbone import HRTBackbone
#TODO: remove if we choose not to have an hrnet directory
from hrnet.pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
#from lib.models.backbones.swin.swin_backbone import SwinTransformerBackbone
from lib.utils.tools.logger import Logger as Log

# TODO: convert into a function that takes a Config object rather than a class.
class BackboneSelector(object):
    def __init__(self, configer):
        # CWDE: Unlike the HRTransformer repository, configer here is a Config object not a Configer object. 
        self.configer = configer

    def get_backbone(self, **params):
        
        # CWDE
        # backbone now receives the option in config.net['BACKBONE'] through a Config object.
        #backbone = self.configer.get("network", "backbone")
        backbone = self.configer.net['BACKBONE']

        model = None

        if backbone == 'hrt':
            # CWDE
            # TODO: alter HRTBackbone and its constructor to accept a Config object with args for construction.
            # Will probably have to wait on the Transformer-Segmentation team to deliver
            model = HRTBackbone(self.configer)(**params)

        elif backbone == 'hrnet':
            model = model = SegmentationNetModule(
            config=config, wandb_run=wandb_run
            )

        else:
            Log.error("Backbone {} is invalid.".format(backbone))
            exit(1)

        return model
