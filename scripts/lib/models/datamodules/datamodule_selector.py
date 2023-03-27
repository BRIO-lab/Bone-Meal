from lib.models.datamodules.datamodules import SegmentationDataModule

class DataModuleSelector():
    def __init__(self, config):
        self.config = config
        
    def get_datamodule(self):
        choice = self.config.net['DATA_MODULE']
        
        data_module = None
        
        if choice == 'segmentation_data_module':
            data_module = SegmentationDataModule(config = self.config)
            
        
        return data_module