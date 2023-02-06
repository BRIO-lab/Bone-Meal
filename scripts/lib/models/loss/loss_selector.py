from lib.utils.tools.logger import Logger as Log
from loss import RMILoss, SegFixLoss, FSAuxRMILoss,FSAuxCELoss, FSAuxOhemCELoss, FSOhemCELoss, FSCELoss, WeightedFSOhemCELoss, OhemCrossEntropy
class LossSelector(object):
    def __init__(self, config, module):
        self.config = config
        self.module = module

    def get_loss(self, **params):
        loss = None
        func = config.config[module]['LOSS']
        if func == 'torch_nn_bce_with_logits_loss':
            loss = torch.nn.BCEWithLogitsLoss()
        elif func == 'fsce_loss':
            loss = FSCELoss(configer = None)
        elif func == 'ohem_ce_loss':
            loss = OhemCrossEntropy(
                ignor_label = config.ohem_ce_loss['IGNORE_LABEL'], thres = config.ohem_ce_loss['THRES'],
                min_kept = config.ohem_ce_loss['MIN_KEPT'], weight = config.ohem_ce_loss['WEIGHT'])
             
        return loss