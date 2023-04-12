from lib.models.datamodules.datamodules import SegmentationDataModule

"""
DataModuleSelector is a class for selecting the appropriate data module for the current task.
Presently, there is only a single DataModule class. Others may be added and specified via the
'DATA_MODULE' field of the config.net dict.
"""
class DataModuleSelector():
    def __init__(self, config):
        self.config = config
        
    def get_datamodule(self):
        choice = self.config.net['DATA_MODULE']
        
        data_module = None
        
        if choice == 'segmentation_data_module':
            data_module = SegmentationDataModule(config = self.config)
            
        
        return data_module