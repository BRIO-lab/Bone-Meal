import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time
import os

"""
ResNet-based Keypoint Estimator
"""
class Configuration:
    def __init__(self):
        """
        self.temp = {
            'train_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/train_3_2_22_fem.csv',
            'val_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/val_3_2_22_fem.csv',
            'test_data': '/home/sasank/Documents/GitRepos/Sasank_JTML_seg/data/3_2_22_fem/test_3_2_22_fem.csv'
        }
        """
        self.init = {
            'PROJECT_NAME': 'Keypoint Estimation',
            'MODEL_NAME': 'Fem_64KP',
            'RUN_NAME': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'WANDB_RUN_GROUP': 'Local',
            'FAST_DEV_RUN': True,  # Runs inputted batches (True->1) and disables logging and some callbacks
            'MAX_EPOCHS': 1,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': None    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
        }
        self.etl = {
            'RAW_DATA_FILE': -1,
            'DATA_DIR': "data",
            # Lol what is this?
            'KEYPOINT_DIRECTORY': "keypoints",
            'KEYPOINT_TXT_FILES': ['tib_KPlabels_16.txt'],
            'VAL_SIZE':  0.2,       # looks sus
            'TEST_SIZE': 0.01,      # I'm not sure these two mean what we think
            #'random_state': np.random.randint(1,50)
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,

            'CUSTOM_TEST_SET': False,
            'TEST_SET_NAME': '/my/test/set.csv'
        }

        self.dataset = {
            'DATA_NAME': 'Ten_Dogs_64KP',
            'SUBSET_PIXELS': True,
            'IMAGE_HEIGHT': 1024,
            'IMAGE_WIDTH': 1024,
            'MODEL_TYPE': 'tib',        # how should we do this? not clear this is still best...
            'CLASS_LABELS': {0: 'bone', 1: 'background'},
            'NUM_KEY_POINTS': 64,
            'IMG_CHANNELS': 1,      # Is this different from self.module['NUM_IMAGE_CHANNELS']
            'STORE_DATA_RAM': False,
            'IMAGE_THRESHOLD': 0,
            'USE_ALBUMENTATIONS': False,

            # What do these do?
            'NUM_PRINT_IMG' : 1,
            'KP_PLOT_RAD' : 3,

            #'NUM_POINTS' : 128,

            'GAUSSIAN_STDDEV' : 5,
            'GAUSSIAN_AMP' : 1e3,

            'STORE_DATA_RAM' : False,

            'CROP_IMAGES' : False,
            'CROP_MIN_X' : 0.29,
            'CROP_MAX_X' : 0.84,
            'CROP_MIN_Y' : 0.45,
            'CROP_MAX_Y' : 0.95,
            
            'IMAGES_PER_GRID': 1,
            'per_grid_image_count_height' : 1, 
            'per_grid_image_count_width' : 1
        }

        """
        # segmentation_net_module needs to be below dataset because it uses dataset['IMG_CHANNELS']
        self.keypoint_net_module = {
            'NUM_KEY_POINTS': 128,
            'NUM_IMG_CHANNELS': self.dataset['IMG_CHANNELS']
        }
        """

        self.datamodule = {
            'IMAGE_DIRECTORY': '/media/sasank/LinuxStorage/Dropbox (UFL)/Canine Kinematics Data/TPLO_Ten_Dogs_grids/',
            'CKPT_FILE': '~/Documents/GitRepos/Stifle-Keypoints/checkpoints/Tib64_200Epochs.ckpt',
            'USE_NAIVE_TEST_SET': False,
            'BATCH_SIZE': 1,
            'SHUFFLE': True,        # Only for training; for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 2,
            'PIN_MEMORY': False
            #'SUBSET_PIXELS': True - this is now in dataset
        }


        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }

        #self.transform = None
        self.transform = \
        A.Compose([
            # Let's do only rigid transformations for now
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.Transpose(p=0.2),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        p=1.0)
