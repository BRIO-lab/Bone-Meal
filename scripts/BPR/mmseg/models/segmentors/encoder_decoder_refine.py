import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from BPR.mmseg.core import add_prefix
from BPR.mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
import pytorch_lightning as pl

@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder,pl.LightningModule):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_coarse_mask=False,      # whether to use coarse mask as input
                 output_float=False,        # whether to return float instead of binary mask
            ):
        super(EncoderDecoderRefine, self).__init__(
            backbone,
            decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.use_coarse_mask = use_coarse_mask
        self.output_float = output_float

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg, coarse_mask):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.use_coarse_mask:
            coarse_mask = (coarse_mask - 0.5) / 0.5
            img = torch.cat([img, coarse_mask[:,None,...]], dim=1)
        x = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

        def training_step(self, train_batch, batch_idx):
            training_batch, training_batch_labels = train_batch['image'], train_batch['label']
            x = training_batch
            print(x.size())

            print("Training batch is on device " + str(x.get_device()))  # testing line
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            crop_size = (128, 128)
            img_metas = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg']),
            ]
            training_output = self.forward_train(img=x, img_metas=img_metas, gt_semantic_seg=training_batch_labels,
                                                 coarse_mask=training_batch_labels)
            # for batch,batch_labels in zip(x,training_batch_labels):
            #     temp2=0
            #     for images,labels in zip(batch,batch_labels):
            #         temp=0
            #         for img,label in zip(images,labels):
            #             print(img.cpu())
            #             print(label)
            #             loss = self.forward_train(self, img=img.cpu(), img_metas=img_metas,
            #                                          gt_semantic_seg=label)
            #             temp+=loss
            #         training_output+=temp/len(images)
            #     training_output+=temp2/len(batch)
            # training_output=training_output/len(x)
            # self.log('exp_train/loss', loss, on_step=True)
            # self.wandb_run.log('train/loss', loss, on_step=True)
            # self.wandb_run.log({'train/loss': loss})
            # self.log(name="train/loss", value=loss)
            return training_output
    def simple_test(self, img, img_meta, coarse_mask, rescale=True):
        if self.use_coarse_mask:
            coarse_mask = (coarse_mask[0] - 0.5) / 0.5
            img = torch.cat([img, coarse_mask[:,None,...]], dim=1)
        # res = super().simple_test(img, img_meta, rescale)

        seg_logit = self.inference(img, img_meta, rescale)
        if self.output_float:
            seg_pred = seg_logit[:,1,:,:]
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred
