##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com, furao17@mails.ucas.ac.cn
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Altered by CWDE615 on 3-3-23 to remove dependency on Configer objects, which do not appear in the Bone-Meal model builder.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper
from lib.models.backbones.hrt.hrt_backbone import HRTBackbone
from lib.models.backbones.hrt.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module

"""
Zach Curran
Rationale: Contains the HRT_SMALL class definition which can be chosen as the backbone for HRT architecture
Future: Alter segmentation head and forward pass to be able to perform segmentation where pixels can belong to multiple classes
"""
class HRT_SMALL_OCR_V2(nn.Module):
    def __init__(self, config):
        super(HRT_SMALL_OCR_V2, self).__init__()
        self.config = config
        self.num_classes = 2 # CWDE: May need to change, but a pixel is either is in the bone or is not for our data. Changed from self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = HRTBackbone(config)()

        in_channels = 480
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type="torchbn"), # TODO: CWDE: replace hardcoding with actual config query, or replace ModuleHelper call with just BatchNorm2d layer
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type="torchbn",
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type="torchbn"),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def segmentation_head(self, input):
        """
        Zach Curran
        Inputs: Output of the default HRT forward pass (tensor)
        Outputs:  Per class pixel probability prediction
        Rationale: Contains the HRT_SMALL class definition which can be chosen as the backbone for HRT architecture
        Future:  Alter to allow segmentation with pixels in multiple classes
        """
        sig = torch.nn.Sigmoid()
        res = sig(input)
        res = res[:, 0, :, :]
        #res = torch.max(res, dim = 1)[0]
        #res = torch.where((res[0].cuda() > 0.5) & (res[1].cuda() == 1), 1, 0)
        res = torch.unsqueeze(res, dim=1)
        return res


    def forward(self, x_):
        """
        Zach Curran
        Inputs: Input tensor
        Outputs: Output tensor passed through segmentation_head which provides prediction
        Rationale: Contains the forward pass for the HRT backbone
        Future: Alter to allow segmentation with pixels in multiple classes
        """
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        #return out_aux, out
        return self.segmentation_head(out)


"""
Zach Curran
Rationale: Contains the HRT_BASE class definition which can be chosen as the backbone for HRT architecture
Future: Add the segmentation head / forward pass from HRT_SMALL_V2 as only HRT_SMALL_V2 was altered for initial testing
"""
class HRT_BASE_OCR_V2(nn.Module):
    def __init__(self, configer):
        super(HRT_BASE_OCR_V2, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 1170
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


"""
Zach Curran
Rationale: Contains the HRT_SMALL_V3 class definition which can be chosen as the backbone for HRT architecture
Future: Add the segmentation head / forward pass from HRT_SMALL_V2 as only HRT_SMALL_V2 was altered for initial testing
"""
class HRT_SMALL_OCR_V3(nn.Module):
    def __init__(self, configer):
        super(HRT_SMALL_OCR_V3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 480
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=hidden_dim,
            key_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
            nn.Conv2d(
                hidden_dim,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out


"""
Zach Curran
Rationale: Contains the HRT_BASE_V3 class definition which can be chosen as the backbone for HRT architecture
Future: Add the segmentation head / forward pass from HRT_SMALL_V2 as only HRT_SMALL_V2 was altered for initial testing
"""
class HRT_BASE_OCR_V3(nn.Module):
    def __init__(self, configer):
        super(HRT_BASE_OCR_V3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 1170
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=hidden_dim,
            key_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
            ),
            ModuleHelper.BNReLU(
                hidden_dim, bn_type=self.configer.get("network", "bn_type")
            ),
            nn.Conv2d(
                hidden_dim,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out
