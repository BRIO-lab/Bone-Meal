"""
Sasank Desaraju
9/14/22
"""

"""
*** This actually does NOT have any Lightning in it. (-_-) ***
It is just a standard Pytorch dataset. I still created this newly to
remake it intentionally."""

import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import cv2
import os


class LitJTMLDataset(Dataset):
    def __init__(self, config, dataset, img_dir):

        self.config = config
        self.transform = self.config.transform
        #self.img_dir = '../../../ImageSegmentation/Img/TPLO_Ten_Dogs_grids'
        self.img_dir = img_dir
        #self.img_dir = '/blue/banks/sasank.desaraju/Sasank_JTML_seg/Images/TPLO_Tend_Dogs_grids_2_22_22'
        #self.data_dir = ''  # I don't know if this will actually get used if I pass in a loaded dataset

        self.dataset = dataset
        self.images = self.dataset[1:,0]
        self.length = len(self.images)

        #if os.path.isfile('/blue/banks/sasank.desaraju/Sasank_JTML_seg/Images/TPLO_Ten_Dogs_grids_2_22_22/grid_Calib file test_000000000381.tif') ==False:
        #    raise Exception('Error, cannot find the first file')

        # image check
        #print('Image directory: ' + self.config.data_constants["IMAGE_DIRECTORY"])
        for idx in range(0,len(self.images)):
            if os.path.isfile(self.img_dir + '/' + self.images[idx]) ==False:
                raise Exception('Error, cannot find file: ' + self.images[idx])
        
        #print(self.config.data_constants['MODEL_TYPE'])
        for i,j in enumerate(self.dataset[0,:]):
            #if j == self.config.data_constants['MODEL_TYPE']:
            if j == 'fem':
                self.labels = self.dataset[1:,i]
        
        # label check
        for idx in range(0,len(self.labels)):
            if os.path.isfile(self.img_dir + '/' + self.labels[idx]) ==False:
                raise Exception('Error, cannot find file: ' + self.labels[idx])
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = io.imread(self.img_dir + '/' + self.images[idx], as_gray = True)
        label = io.imread(self.img_dir + '/' + self.labels[idx], as_gray = True)        
    
        label_dst = np.zeros_like(label)
        label_normed = cv2.normalize(label, label_dst, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
        label = label_normed

    # SUBSET_PIXEL
	# This constrains the bounds of the image to just around the bone
        if self.config.datamodule["SUBSET_PIXELS"] == True:
            kernel = np.ones((30,30), np.uint8)
            label_dilated = cv2.dilate(label, kernel, iterations = 5)
            image_subsetted = cv2.multiply(label_dilated, image)
            image = image_subsetted
        
        if self.config.dataset['USE_ALBUMENTATIONS'] == True:
            image = np.array(image)
            label = np.array(label)
            transformed = self.transform(image = image, mask = label)
            image = transformed["image"]
            label = transformed["mask"]

        
        image = torch.FloatTensor(image[None, :, :])
        label = torch.FloatTensor(label[None, :, :])
        img_name = self.images[idx]
        sample = {'image': image, 'label': label, 'img_name': img_name}


        return sample
