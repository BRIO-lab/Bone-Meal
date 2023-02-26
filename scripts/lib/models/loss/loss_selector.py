from lib.models.loss.loss import RMILoss, SegFixLoss, FSAuxRMILoss,FSAuxCELoss, FSAuxOhemCELoss, FSOhemCELoss, FSCELoss, WeightedFSOhemCELoss, OhemCrossEntropy
import sys
import os
import torch.nn
from importlib import import_module

class LossSelector(object):
    def __init__(self, config, module_dict):
        self.config = config
        self.module_dict = module_dict
        
    def get_loss(self, **params):
        loss = None
        func = self.module_dict['LOSS']
        
        if func == 'torch_nn_bce_with_logits_loss':
            loss = torch.nn.BCEWithLogitsLoss()
        elif func == 'fsce_loss':
            loss = FSCELoss(configer = None)
        elif func == 'ohem_ce_loss':
            loss = OhemCrossEntropy(
                ignore_label = self.config.ohem_ce_loss['IGNORE_LABEL'], thres = self.config.ohem_ce_loss['THRES'],
                min_kept = self.config.ohem_ce_loss['MIN_KEPT'], weight = self.config.ohem_ce_loss['WEIGHT'])
            
        return loss