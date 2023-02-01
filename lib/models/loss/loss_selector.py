from lib.utils.tools.logger import Logger as Log
from loss import RMILoss, SegFixLoss, FSAuxRMILoss,FSAuxCELoss, FSAuxOhemCELoss, FSOhemCELoss,FSCELoss, WeightedFSOhemCELoss
class LossSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def get_loss(self, **params):
        loss = None
        func=self.configer.get("network","loss")
        if "hrt" in func:
            loss = None

        elif "hrnet" in func:
            loss = None

        else:
            Log.error("Loss function {} is invalid.".format(func))
            exit(1)

        return loss