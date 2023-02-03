"""
Sasank Desaraju
9/9/2022
"""

#from asyncio.log import logger
from datetime import datetime
from importlib import import_module
from unicodedata import name
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule import SegmentationDataModule
from callbacks import JTMLCallback
from utility import create_config_dict
#import click
import sys
import os
import time
import wandb
from build import build_model 
# want to refactor more


"""
The main function contains the neural network-related code.
"""
def main(config, wandb_run):

    # The DataModule object loads the data from CSVs, calls the JTMLDataset to get data, and creates the dataloaders.
    data_module = SegmentationDataModule(config=config)

    # This is the real architecture we're using. It is vanilla PyTorch - no Lightning.
    #pose_hrnet = PoseHighResolutionNet(num_key_points=1, num_image_channels=config.module['NUM_IMAGE_CHANNELS'])
    
    # This is our LightningModule, which where the architecture is supposed to go.
    # Since we are using an architecure written in PyTorch (PoseHRNet), we feed that architecture in.
    # We also pass our wandb_run object so we can log.
    #model = SegmentationNetModule(config=config, wandb_run=wandb_run) # I can put some data module stuff in this argument if I want
    """
    Call to Build is made here
    modified by Engut (the one presently writing this)
    """
    model = build_model(config=config, wandb_run=wandb_run)
    # This is a callback that should help us with stopping validation when it's time but isn't working.
    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                        mode='min',
                                                        dirpath='checkpoints/',
                                                        filename=wandb_run.name)
                                                        #filename=wandb.run.name)

    # Our trainer object contains a lot of important info.
    trainer = pl.Trainer(
        # If the below line is an error, change it to cpu and 1 device
        accelerator='gpu',
        # accelerator='cpu',
        devices=-1,     # use all available devices (GPUs)
        # devices=1,
        auto_select_gpus=True,  # helps use all GPUs, not quite understood...
        #logger=wandb_logger,   # tried to use a WandbLogger object. Hasn't worked...
        default_root_dir=os.getcwd(),
        callbacks=[JTMLCallback(config, wandb_run)],    # pass in the callbacks we want
        #callbacks=[JTMLCallback(config, wandb_run), save_best_val_checkpoint_callback],
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'])
    
    # This is the step where everything happens.
    # Fitting includes both training and validation.
    trainer.fit(model, data_module)

    # TODO: Are the trainer and Wandb doing the same thing/overwriting the checkpoint?
    #Save model using .ckpt file format. This includes .pth info and other (hparams) info.
    trainer.save_checkpoint(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + config.init['MODEL_NAME'] + '.ckpt')
    
    # Save model using Wandb
    wandb.save(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + '/' + config.init['MODEL_NAME'] + '.ckpt')
    wandb_run.config.update({'Model Save Directory': CKPT_DIR + config.init['WANDB_RUN_GROUP'] + '/' + config.init['MODEL_NAME'] + '.ckpt'})

if __name__ == '__main__':
    

    ## Setting up the config
    # Parsing the config file
    CONFIG_DIR = os.getcwd() + '/config/'
    sys.path.append(CONFIG_DIR)

    # CWDE
    # load config file. Argument one should be the name of the file without the .py extension.
    config_module = import_module(sys.argv[1])
    
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
        job_type='fit',                         # Lets us know in Wandb that this was a fit run
        config=create_config_dict(config)
        #id=str(time.strftime('%Y-%m-%d-%H-%M-%S'))     # this can be used for custom run ids but must be unique
        #dir='logs/'
        #save_dir='/logs/'
    )
    #wandb_placeholder = wandb.init()

    main(config, wandb_run)
    #main(config, wandb_placeholder)

    # Sync and close the Wandb logging. Good to have for DDP, I believe.
    wandb.finish()
