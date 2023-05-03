"""
Sasank Desaraju
9/23/22
This is to handle callbacks to keep our code clean and nice."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
import math
from utility import run_metrics


class JTMLCallback(Callback):
    def __init__(self, config, wandb_run) -> None:
        super().__init__()

        self.config = config
        self.wandb_run = wandb_run
        #self.wandb_logger = wandb_logger
        self.min_val_loss = math.inf
        self.class_labels = self.config.dataset['CLASS_LABELS']

    """
    *********************** Init ***********************
    """

    """
    *********************** Fit ***********************
    """

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Fitting!  " + 20 * '*')
        return super().on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Fitting!  " + 20 * '*')
        return super().on_fit_end(trainer, pl_module)
    
    """
    *********************** Fit/Train ***********************
    """

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Training!  " + 20 * '*')
        self.wandb_run.log({'INFO': 'Starting Training!'})
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Training!  " + 20 * '*')
        self.wandb_run.log({'INFO': 'Finished Training!'})
        return super().on_train_end(trainer, pl_module)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('\n' + 20 * '*' + f'Starting train epoch {pl_module.current_epoch}!' + 20 * '*' + '\n')
        #return super().on_epoch_start(trainer, pl_module)
        return

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('\n' + 20 * '*' + f'Finished train epoch {pl_module.current_epoch}!' + 20 * '*' + '\n')
        #return super().on_epoch_end(trainer, pl_module)
        return

    """
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_batch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_batch_end(trainer, pl_module)
    """

    """
    *********************** Fit/Validation ***********************
    """

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('\n' + 20 * '*' + "  Starting Validation!  " + 20 * '*' + '\n')
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('\n' + 20 * '*' + "  Finished Validation!  " + 20 * '*' + '\n')
        # TODO: is the below line right? Should it be in this hook or in on_validation_epoch_end()?
        print(15 * '*' + 'Min Validation Loss is ' + str(self.min_val_loss))
        return super().on_validation_end(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #self.wandb_run.log
        return super().on_validation_epoch_end(trainer, pl_module)

    """
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    """
    
    """
    Zach Curran

    Inputs: Current validation batch

    Outputs: Logs validation batch info to wandb

    Rationale: Used to help visualize validation results

    Future: Create a parameter to determine whether or not to log images

    """
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        # TODO: Is outputs always just 1 loss value? If not, this might break bc it assumes outputs is comparable to min_val_loss
        if outputs.item() < self.min_val_loss:
            self.min_val_loss = outputs.item()
        
        self.wandb_run.log({'validation/loss': outputs.item()})
        self.log("val_loss", outputs.item())

        val_outputs = pl_module.forward(batch['image'])

        #for idx in range(self.config.datamodule['BATCH_SIZE']):
        for idx in range(len(batch['img_name'])):
            
            # Getting the batch's data
            img_name = batch['img_name'][idx]
            input_image = batch['image'][idx][0]
            label_image = batch['label'][idx][0]
            output_image = torch.round(val_outputs[idx][0])
            
            # Inputs
            wandb_input = wandb.Image(input_image, caption=img_name)
            # TODO: Should I make this 'validation/epoch_idx_' + epoch_idx + '/input_image' ?
            # CWDE: commented out to save space on hpg. Consider parameterizing the decision to log this in the future.
            # self.wandb_run.log({'validation/input_image': wandb_input})
            
            # Labels
            wandb_label = wandb.Image(label_image, caption=img_name)
            # CWDE: commented out to save space on hpg. Consider parameterizing the decision to log this in the future.
            # self.wandb_run.log({'validation/input_label': wandb_label})
            
            # Predictions
            wandb_output = wandb.Image(output_image, caption=img_name)
            # CWDE: commented out to save space on hpg. Consider parameterizing the decision to log this in the future.
            # self.wandb_run.log({'validation/output_image': wandb_output})
            
            # Overlay Image
            # CWDE: commented out to save space on hpg. Consider parameterizing the decision to log this in the future.
            '''
            self.wandb_run.log(
                {'validation/overlay': wandb.Image(input_image,
                caption=img_name,
                masks={
                    'predictions': {
                        'mask_data': output_image.detach().cpu().numpy(),
                        'class_labels': self.class_labels
                    },
                    'ground_truth': {
                        # the mask_data here is actually on CPU since it needs to be numpy which is only on CPU
                        'mask_data': label_image.detach().cpu().numpy(),
                        'class_labels': self.class_labels
                    }
                })}
            )
            '''
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)



    """
    *********************** Test ***********************
    """

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Testing!  " + 20 * '*')
        self.wandb_run.log({'INFO': 'Starting Testing!'})
        return super().on_test_start(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Testing!  " + 20 * '*')
        self.wandb_run.log({'INFO': 'Finished Testing!'})
        return super().on_test_end(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + f'Finished test epoch {pl_module.current_epoch}!' + 20 * '*')
        return super().on_test_epoch_end(trainer, pl_module)

    #def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        #return super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    """
    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int) -> None:
        return super().on_test_batch_start(trainer, pl_module, batch, batch_idx)
    """

    """
    Zach Curran

    Inputs: Current test batch

    Outputs: Logs test batch info to wandb

    Rationale: Used to help visualize test results

    Future: Create a parameter to determine whether or not to log images

    """
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx) -> None:
        self.wandb_run.log({'test/loss': outputs.item()})

        test_outputs = pl_module.forward(batch['image'])

        #for idx in range(self.config.datamodule['BATCH_SIZE']):
        for idx in range(len(batch['img_name'])):
            
            # Getting the batch's data
            img_name = batch['img_name'][idx]
            input_image = batch['image'][idx][0]
            label_image = batch['label'][idx][0]
            output_image = torch.round(test_outputs[idx][0])

            # Inputs
            wandb_input = wandb.Image(input_image, caption=img_name)
            self.wandb_run.log({'test/input_image': wandb_input})
            
            # Labels
            wandb_label = wandb.Image(label_image, caption=img_name)
            self.wandb_run.log({'test/input_label': wandb_label})
            
            # Formatting for image. Depending on the Architecture saved in the ckpt, extra image processing may be necessary before conversion
            # into a WandB Image object. Do this by adding a case for your architecture to the if ladder here, and make sure to specify 
            # net.'ARCHITECTURE' correctly.
            proc_output_image = None

            if self.config.net['ARCHITECTURE'] == 'seg_hrnet':
                proc_output_image = (output_image > 0).type(torch.float32) # safeguard against models that leave background negative.
            else:
                proc_output_image = output_image

            # Predictions
            wandb_output = wandb.Image(proc_output_image, caption=img_name)
            self.wandb_run.log({'test/output_image': wandb_output})
            
            # Overlay Image
            self.wandb_run.log(
                {'test/overlay': wandb.Image(input_image,
                caption=img_name,
                masks={
                    'predictions': {
                        'mask_data': output_image.detach().cpu().numpy(),
                        'class_labels': self.class_labels
                    },
                    'ground_truth': {
                        # the mask_data here is actually on CPU since it needs to be numpy which is only on CPU
                        'mask_data': label_image.detach().cpu().numpy(),
                        'class_labels': self.class_labels
                    }
                })}
            )

        # Logging metrics
        metric_dict = run_metrics(output_image, label_image, self.config.dataset['IMAGE_THRESHOLD'])
        self.wandb_run.log(metric_dict)

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)