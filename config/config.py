import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time
import os

class Configuration:
    def __init__(self):
        # CWDE: self.init MUST BE LISTED FIRST
        self.init = {
            'PROJECT_NAME': 'Segmentation Trial',
            'MODEL_NAME': 'MyModel',
            'RUN_NAME': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'WANDB_RUN_GROUP': 'Local',
            'FAST_DEV_RUN': False,  # Runs inputted batches (True->1) and disables logging and some callbacks
            'MAX_EPOCHS': 25,
            'MAX_STEPS': -1,    # -1 means it will do all steps and be limited by epochs
            'STRATEGY': None,    # This is the training strategy. Should be 'ddp' for multi-GPU (like HPG)
            'PATIENCE': 3,     # Stop training after this many validation checks without improvement
            'STOPPING_THRESHOLD': 0.01,     # Stop training when val loss reaches this threshold
            'VAL_CHECK_INTERVAL': 1     # Perform validation check after this many epochs
        }
        self.etl = {
            'RAW_DATA_FILE': -1,    # -1 means it will create a full data csv from the image directory, using all images in the image directory
            #'RAW_DATA_FILE': 'my_data.csv',
            'DATA_DIR': "data",
            'VAL_SIZE':  0.1,       # looks sus
            'TEST_SIZE': 0.1,      # I'm not sure these two mean what we think
            #'random_state': np.random.randint(1,50)
            # HHG2TG lol; deterministic to aid reproducibility
            'RANDOM_STATE': 42,

            'CUSTOM_TEST_SET': False,
            'TEST_SET_NAME': '/my/test/set.csv'
        }

        self.dataset = {
            'DATA_NAME': 'Ten_Dogs_64KP',
            'IMAGE_HEIGHT': 1024,
            'IMAGE_WIDTH': 1024,
            'MODEL_TYPE': 'tib',        # specifies that it's a femur model. how should we do this? not clear this is still best...
            'CLASS_LABELS': {0: 'bone', 1: 'background'},
            'IMG_CHANNELS': 1,      # Is this differnt from self.module['NUM_IMAGE_CHANNELS']
            'IMAGE_THRESHOLD': 0,
            'SUBSET_PIXELS': True,
            'USE_ALBUMENTATIONS': True,
            'NUM_KEY_POINTS' : 64,
        }

        self.datamodule = {
            # *** CHANGE THE IMAGE DIRECTORY TO YOUR OWN ***
            #'IMAGE_DIRECTORY': '/media/sasank/LinuxStorage/Dropbox (UFL)/Canine Kinematics Data/TPLO_Ten_Dogs_grids',
            
            # Z. Curran:  '/blue/banks/TPLO_Ten_Dogs_grids'
            # CWDE: "C:/Users/cwell/Documents/jtml_data/TPLO_Ten_Dogs_grids"
            # CWDE: '/home/driggersellis.cw/jtml_data/TPLO_Ten_Dogs_grids/' 
            
            'IMAGE_DIRECTORY': '/home/driggersellis.cw/jtml_data/TPLO_Ten_Dogs_grids/',

            # *** CHANGE THE CHECKPOINT PATH TO YOUR OWN FOR TESTING ***
            #'CKPT_FILE': 'path/to/ckpt/file.ckpt',  # used when loading model from a checkpoint
            # used when loading model from a checkpoint, such as in testing
            
            # Z. Curran : '/blue/banks/curran.z/Bone-Meal/checkpoints/'
            # CWDE: "C:/Users/cwell/Documents/jtml_data/Checkpoints/"
            # CWDE: '/home/driggersellis.cw/Bone-Meal/checkpoints/' 
            
            'CKPT_FILE': '/home/driggersellis.cw/Bone-Meal/checkpoints/' + self.init['WANDB_RUN_GROUP'] + self.init['MODEL_NAME'] + '.ckpt', 
            'BATCH_SIZE': 4,
            'SHUFFLE': False,        # Only for training, for test and val this is set in the datamodule script to False
            'NUM_WORKERS': 4,   # This number seems fine for local but on HPG, we have so many cores that a number like 4 seems better.
            'PIN_MEMORY': False,
            #'SUBSET_PIXELS': True,
            'USE_NAIVE_TEST_SET': False
        }
    

        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }
        
        
        # network params
        self.net = {
            # 'hrt_small', 'hrnet'
            'BACKBONE': 'hrt_small', # the name of the backbone identified in backbone_selector. Currently have planned support for hrt and hrnet
            # 'seg_hrt', 'seg_hrnet'
            'ARCHITECTURE' :'seg_hrt', # name of the architecture_builder class file
            'DATA_MODULE' : 'segmentation_data_module'
        }
        
        # PARAMS FOR BACKBONES (Format: self.[name of backbone in self.net] = { params dict })
        
        # these are essentially params for the hrnet backbone's SegmentationNetModule class
        # they are an exception to the format established in the comment above.
        self.segmentation_net_module = {
                'NUM_KEY_POINTS' : 1,
                'NUM_IMG_CHANNELS': self.dataset['IMG_CHANNELS'],
                'LOSS' : 'torch_nn_bce_with_logits_loss'
        }
        
        # Params for HRT's segmentation_net_module. Defaults used from HRT's Base config
        self.hrt_segmentation_net = {
                'MODEL_CONFIG' : 'hrt_small',
                'LOSS' : 'torch_nn_bce_loss'
        }
        
        # PARAMS FOR LOSS FUNCTIONS (Format: self.[name of loss in self.backbone] = { params dict })
        
        # Params dict for BCEWithLogitsLoss, which takes no params in the origin model from Lightning Segmentation.
        self.torch_nn_bce_with_logits_loss = {
            # NO PARAMS
        }

        self.torch_nn_bce_loss = {
            # NO PARAMS
        }
        
        self.ohem_ce_loss = {
            'IGNORE_LABEL' : -1,
            'THRES' : 0.7,
            'MIN_KEPT' : 100000,
            'WEIGHT' : None
        }
        
        # Params for FSCELoss: TODO: insert actual params
        self.fsce_loss = {
            'ce_weight' : -1,
            'ce_reduction' : -1,
            'ce_ignore_index': -1
        }
        
        #TODO: add other params dicts for each loss function we have. Code will be extensible 

        # Commented out transforms do not support keypoints
        self.transform = \
        A.Compose([
        # A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
        A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
        # A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
        A.Flip(always_apply=False, p=0.5),
        # A.ElasticTransform(always_apply=False, p=0.85, alpha=0.5, sigma=150, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
        # A.InvertImg(always_apply=False, p=0.5),
        # A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
        # A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    p=0.85)
