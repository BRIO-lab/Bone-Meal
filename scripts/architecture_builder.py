from torch import nn
from config import config
import torch.optim as optim
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.loss.loss_selector import LossSelector

class ArchitectureBuilder(nn.Module):
    def __init__(self, config):
        self.config=config
    def build_model(self):
        Backbone_selector= BackboneSelector(self.config)
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
# CWDE: TODO: Determine how to do this with a module_dict instead of just config, which can be None in LossSelector.get_loss,
#             provided that it is not needed in the construction of the loss function through its param dict
    def compile_model(self,model):
        Loss_selector= LossSelector(self.config)
        loss_fn = Loss_selector.get_loss()
        #add wherever the learning rate is stored in config. If not stored, put hardcoded value
        optimizer = optim.Adam(model.parameters(), lr=self.config[''])
        model.compile(loss=loss_fn, optimizer=optimizer)
        return model