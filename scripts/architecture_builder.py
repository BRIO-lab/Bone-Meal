from torch import nn
from config import config
import torch.optim as optim
from lib.models.backbones import backbone_selector
from lib.models.loss import loss_selector

class ArchitectureBuilder(nn.Module):
    def __init__(self, config):
        self.config=config
    def build_model(self):
        Backbone_selector= backbone_selector(self.config.net['BACKBONE'])
        backbone = Backbone_selector.get_backbone()
        model = nn.sequential(backbone)
        #Following code is for if we have a module selector with multiple modules
        #The following code is basically to select modules from multiple entries in the dict
        # for module_config in self.config['modules']:
        #     Module_selector=module_selector(module_config['name of module'])
        #     module=Module_selector.get_module()
        #     model.add_module(module_config['name of module'])

        return model

##No forward function as those would be defined in the modules.
    def compile_model(self,model):
        Loss_selector=loss_selector(self.config[''])
        loss_fn = Loss_selector.get_loss()
        #add wherever the learning rate is stored in config. If not stored, put hardcoded value
        optimizer = optim.Adam(model.parameters(), lr=self.config[''])
        model.compile(loss=loss_fn, optimizer=optimizer)
        return model