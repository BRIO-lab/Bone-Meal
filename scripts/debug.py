from lib.models.datamodules.datamodule_selector import DataModuleSelector
from importlib import import_module
import sys
import os

CONFIG_DIR = os.getcwd() + '/config/'
sys.path.append(CONFIG_DIR)

config_module = import_module(sys.argv[1])
config = config_module.Configuration()

data_selector = DataModuleSelector(config = config)
data_module = data_selector.get_datamodule()

data_module.setup(None)
JTML_data = data_module.training_set

for idx in range(len(JTML_data)) :
    sample = JTML_data.__getitem__(idx)
    kp_label = sample['kp_label']
    kp_raw = sample['kp_raw']
    
    if kp_label.shape[0] != 64:
        print(f"idx : {idx}  len : {kp_label.shape[0]}  raw : {kp_raw}")

