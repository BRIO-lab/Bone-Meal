"""
Sasank Desaraju
9/21/2022
"""

#from asyncio.log import logger
from datetime import datetime
from importlib import import_module
from unicodedata import name
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#from pose_hrnet_module import SegmentationNetModule, PoseHighResolutionNet
from models import SegmentationNetModule, PoseHighResolutionNet
from datamodule import SegmentationDataModule
from callbacks import JTMLCallback
from utility import create_config_dict
#import click
import sys
import os
import time
import wandb

#torch.manual_seed(42) # haha HG2G


"""
The main function contains the neural network-related code.
"""

def main(config, wandb_run):
    # The DataModule object loads the data from CSVs, calls the JTMLDataset to get data, and creates the dataloaders.
    data_module = SegmentationDataModule(
        config=config)
    # This is the real architecture we're using. It is vanilla PyTorch - no Lightning.
    #pose_hrnet = PoseHighResolutionNet(num_key_points=1, num_image_channels=config.module['NUM_IMAGE_CHANNELS'])
    #model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run).load_from_checkpoint(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')
    # This is our LightningModule, which where the architecture is supposed to go.
    # Since we are using an architecure written in PyTorch (PoseHRNet), we feed that architecture in.
    # We also pass our wandb_run object to we can log.
    if config.datamodule['CKPT_FILE'] != None:
        model = SegmentationNetModule.load_from_checkpoint(config.datamodule['CKPT_FILE'], pose_hrnet=pose_hrnet, wandb_run=wandb_run)
        print('Checkpoint file loaded from ' + config.datamodule['CKPT_FILE'])
    elif config.datamodule['CKPT_FILE'] == None:
        try:
            model = SegmentationNetModule.load_from_checkpoint(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + '/' + config.init['MODEL_NAME'] +'.ckpt', pose_hrnet=pose_hrnet, wandb_run=wandb_run)
        except:
            print("No checkpoint .ckpt file in default location at " + CKPT_DIR + config.init['WANDB_RUN_GROUP'] + '/' + config.init['MODEL_NAME'] +'.ckpt')
    #model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run)
    #model = model.load_from_checkpoint(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')

    """
    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                        mode='min',
                                                        dirpath='checkpoints/',
                                                        filename=wandb_run.name)
    """

    # Our trainer object contains a lot of important info.
    trainer = pl.Trainer(
        # If the below line gives an error because you don't have a GPU, then comment it out and uncomment the line after it which uses the CPU.
        accelerator='gpu',
        # accelerator='cpu',
        devices=-1,     # use all available devices (GPUs)
        # Probably comment out the below line if you're using your CPU.
        auto_select_gpus=True,  # helps use all GPUs, not quite understood...
        #logger=wandb_logger,
        default_root_dir=os.getcwd(),
        callbacks=[JTMLCallback(config, wandb_run)],    # pass in the callbacks we want
        #callbacks=[save_best_val_checkpoint_callback],
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'])
    # This is the step where everything happens.
    trainer.test(model, data_module)

if __name__ == '__main__':
    ## Setting up the config
    # Parsing the config
    CONFIG_DIR = os.getcwd() + '/config/'
    sys.path.append(CONFIG_DIR)
    config_module = import_module(sys.argv[1])
    #config_module = import_module('config/config')
    # Instantiating the config file
    config = config_module.Configuration()

    # Setting the checkpoint directory
    CKPT_DIR = os.getcwd() + '/checkpoints/'

    ## Setting up the logger
    # Setting the run group as an environment variable. Mostly for DDP (on HPG)
    os.environ['WANDB_RUN_GROUP'] = config.init['WANDB_RUN_GROUP']

    # Creating the Wandb run object
    wandb_run = wandb.init(
        project=config.init['PROJECT_NAME'],    # Leave the same for the project (e.g. JTML_seg)
        name=config.init['RUN_NAME'],           # Should be diff every time to avoid confusion (e.g. current time)
        group=config.init['WANDB_RUN_GROUP'],
        job_type='test',                         # Lets us know in Wandb that this was a test run
        config=create_config_dict(config)
    )

    main(config,wandb_run)

    # Sync and close the Wandb logging. Good to have for DDP, I believe.
    wandb_run.finish()
