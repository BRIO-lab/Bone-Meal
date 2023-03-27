from lib.models.nets.pose_hrnet_module import SegmentationNetModule as HRNetSegmentationNetModule
from lib.models.nets.seg_hrt import SegmentationNetModule as HRTSegmentationNetModule
from build import build_model


class ArchitectureSelector():
    def __init__(self, config, wandb):
        self.config = config
        self.wandb = wandb
    
    def get_architecture(self):
        choice = self.config.net['ARCHITECTURE']
        
        builder = None
        
        if choice == "seg_hrnet":
            builder = HRNetSegmentationNetModule(self.config, self.wandb)
        elif choice == "seg_hrt":
            builder = HRTSegmentationNetModule(self.config, self.wandb)
        elif not choice: # run just a backbone
            builder = build_model(config=self.config, wandb_run=self.wandb)
            
        return builder