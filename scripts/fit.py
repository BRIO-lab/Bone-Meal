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
from callbacks import JTMLCallback
from utility import create_config_dict
#import click
import sys
import os
import time
import wandb
from build import build_model 
# want to refactor more

# CWDE: 2-23 & 24-2023
from lib.models.datamodules.datamodule_selector import DataModuleSelector
from lib.models.nets.architecture_selector import ArchitectureSelector

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

"""
The main function contains the neural network-related code.
"""
def main(config, wandb_run):

    """
    The DataModule object loads the data from CSVs, calls the JTMLDataset to get data, and creates the dataloaders.
    """
    data_selector = DataModuleSelector(config = config)
    data_module = data_selector.get_datamodule()
    
    
    """
    Call to Build is made inside of Architecture Selector
    """
    model = ArchitectureSelector(config, wandb_run).get_architecture()
    
    """
    Construct callbacks 
    """
    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                                        mode='min',
                                                        dirpath=CKPT_DIR,
                                                        save_top_k = config.init['SAVE_TOP_K'],
                                                        filename=config.init['WANDB_RUN_GROUP'] + config.init['MODEL_NAME'] +'_best')
    
    jtml_callback = JTMLCallback(config, wandb_run)
    

    """
    Construct pl.Trainer. This will train the architecture made by architecture selector.
    """
    # Our trainer object contains a lot of important info.
    trainer = pl.Trainer(
        # If the below line is an error, change it to cpu and 1 device
        accelerator='gpu',      # alternatively: accelerator='cpu',
        devices=-1,             # use all available devices (GPUs)
        # devices=1,
        auto_select_gpus=True,  # helps use all GPUs, not quite understood...
        #logger=wandb_logger,   # tried to use a WandbLogger object. Hasn't worked...
        default_root_dir=os.getcwd(),
        callbacks=[jtml_callback, save_best_val_checkpoint_callback],    # pass in the callbacks we want
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'],
        check_val_every_n_epoch=config.init['VAL_CHECK_INTERVAL'])
        #val_check_interval=config.init['MAX_STEPS'])
    
    """
    Fit model
    """
    # This is the step where everything happens.
    # Fitting includes both training and validation.
    trainer.fit(model, data_module)

    """
    Save final checkpoint
    """
    #Save model using .ckpt file format. This includes .pth info and other (hparams) info.
    trainer.save_checkpoint(CKPT_DIR + config.init['WANDB_RUN_GROUP'] + config.init['MODEL_NAME'] + '_final.ckpt')
    
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

    # WARNING: If you receive Errno 28 on hpg, this means that you are out of memory. You will need to cull local wandb files/ckpts to continue.
    main(config, wandb_run)

    # Sync and close the Wandb logging. Good to have for DDP, I believe.
    wandb.finish()
