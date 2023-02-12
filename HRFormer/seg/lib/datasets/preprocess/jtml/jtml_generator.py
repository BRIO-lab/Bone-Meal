#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Lang Huang(layenhuang@outlook.com)
# JTML aug data generator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import shutil
import scipy.io as sio
import cv2
import numpy as np
import torch
from skimage import io


LABEL_DIR = 'label'
IMAGE_DIR = 'image'


class JTMLGenerator(object):
    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR):
        self.args = args
        self.save_dir = self.args.save_dir

    def generate_label(self):
        img_folder = os.path.join(self.args.ori_root_dir, 'imgs')
        labels_folder = os.path.join(self.args.ori_root_dir, 'labels')

        for curr in os.listdir(labels_folder):
            label = io.imread(curr, as_gray = True)
            print(label.shape)
            label_dst = np.zeros_like(label)
            label_normed = cv2.normalize(label, label_dst, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
            label = label_normed   
            cv2.imwrite(os.path.join(self.save_dir, 'test'), label)

        for filename in os.listdir(img_folder):
            image = io.imread(filename, as_gray = True)
            print("IMAGE SHAPE:", image.shape)  
            cv2.imwrite(os.path.join(self.save_dir, 'test'), label)      


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the jtml data.')

    args = parser.parse_args()

    pcontext_generator = JTMLGenerator(args)
    pcontext_generator.generate_label()