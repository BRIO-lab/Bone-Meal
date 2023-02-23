 from datamodules import SegmentationDataModule

class DataModuleSelector():
    def __init__(self, config):
        self.config = config
        
    def get_data_module(self):
        choice = self.config.net['DATA_MODULE']
        
        data_module = None
        
        if data_module == 'segmentation_data_module':
            SegmentationDataModule(config = self.config)
        
        return data_module