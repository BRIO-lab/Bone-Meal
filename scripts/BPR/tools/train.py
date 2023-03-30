import argparse
import copy
import os
import os.path as osp
import time
import wandb
import pytorch_lightning as pl
from callbacks import JTMLCallback
import sys

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from importlib import import_module

from BPR.mmseg import __version__
from BPR.mmseg.apis import set_random_seed, train_segmentor
from BPR.mmseg.datasets import build_dataset
from BPR.mmseg.models import build_segmentor
from BPR.mmseg.utils import collect_env, get_root_logger

from lib.models.datamodules.datamodule_selector import DataModuleSelector
from utility import create_config_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(config, wandb_run):
    args = parse_args()

    cfg = Config.fromfile(os.getcwd()+'/BPR/configs/bpr/hrnet18s_128.py')
    print(cfg)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # log some basic info

    # set random seeds
    if args.seed is not None:
        # logger.info(f'Set random seed to {args.seed}, deterministic: '
        #             f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # datasets = [build_dataset(cfg.data.train)]
    # logger.info(datasets)
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))
    # if cfg.checkpoint_config is not None:
    #     # save mmseg version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
    #         config=cfg.pretty_text,
    #         CLASSES=datasets[0].CLASSES,
    #         PALETTE=datasets[0].PALETTE)
    # # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    trainer = pl.Trainer(
        # If the below line is an error, change it to cpu and 1 device
        accelerator='gpu',
        # accelerator='cpu',
        devices=-1,  # use all available devices (GPUs)
        # devices=1,
        auto_select_gpus=True,  # helps use all GPUs, not quite understood...
        # logger=wandb_logger,   # tried to use a WandbLogger object. Hasn't worked...
        default_root_dir=os.getcwd(),
        callbacks=[JTMLCallback(config, wandb_run)],  # pass in the callbacks we want
        # callbacks=[JTMLCallback(config, wandb_run), save_best_val_checkpoint_callback],
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'])
    # train_segmentor(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=(not args.no_validate),
    #     timestamp=timestamp,
    #     meta=meta)
    data_selector = DataModuleSelector(config=config)
    data_module = data_selector.get_datamodule()
    trainer.fit(model, data_module)

if __name__ == '__main__':
     ## Setting up the config
    # Parsing the config file
    CONFIG_DIR = 'C:/Users/echen/PycharmProjects/Bone-Meal' + '/config/'
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
