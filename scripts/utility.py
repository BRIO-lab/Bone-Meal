"""
Changed by Sasank Desaraju
9/24/22
"""

import logging
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from collections import OrderedDict

"""
LitJTML (the Lightning stuff really only makes use of the metric stuff at the bottom.
Everything else (parse_config, set_logger, etc) was just copied over but it not being used
in our Lightning workflow.
"""

def create_config_dict(config) -> dict:
    config_dict =   {'init/Project Name': config.init['PROJECT_NAME'],
                    'init/Model Name': config.init['MODEL_NAME'],
                    'init/Run Name': config.init['RUN_NAME'],
                    'init/Wandb Run Group': config.init['WANDB_RUN_GROUP'],
                    'init/Fast Dev Run': config.init['FAST_DEV_RUN'],
                    'init/Max Epochs': config.init['MAX_EPOCHS'],
                    'init/Max Steps': config.init['MAX_STEPS'],
                    'init/Strategy': str(config.init['STRATEGY']),
                    'etl/Raw Data File': config.etl['RAW_DATA_FILE'],
                    'etl/Data Directory': config.etl['DATA_DIR'],
                    'etl/Validation Size': config.etl['VAL_SIZE'],
                    'etl/Test Size': config.etl['TEST_SIZE'],
                    'etl/Random State': config.etl['RANDOM_STATE'],
                    'etl/Custom Test Set': config.etl['CUSTOM_TEST_SET'],
                    'etl/Test Set Name': config.etl['TEST_SET_NAME'],
                    'dataset/Image Height': config.dataset['IMAGE_HEIGHT'],
                    'dataset/Image Width': config.dataset['IMAGE_WIDTH'],
                    'dataset/Model Type': config.dataset['MODEL_TYPE'],
                    'dataset/Class Labels': config.dataset['CLASS_LABELS'],
                    'dataset/Image Channels': config.dataset['IMG_CHANNELS'],
                    'dataset/Image Threshold': config.dataset['IMAGE_THRESHOLD'],
                    'dataset/Using Albumentations': config.dataset['USE_ALBUMENTATIONS'],
                    'datamodule/Image Directory': config.datamodule['IMAGE_DIRECTORY'],
                    'datamodule/Checkpoint File': str(config.datamodule['CKPT_FILE']),
                    'datamodule/Batch Size': config.datamodule['BATCH_SIZE'],
                    'datamodule/Shuffle': config.datamodule['SHUFFLE'],
                    'datamodule/Num Workers': config.datamodule['NUM_WORKERS'],
                    'datamodule/Pin Memory': config.datamodule['PIN_MEMORY'],
                    'datamodule/Subset Pixels': config.datamodule['SUBSET_PIXELS'],
                    #'module/Loss Function': config.module['LOSS_FN'],
                    #'module/Num Image Channels': config.module['NUM_IMAGE_CHANNELS'],
                    'hparams/Load From Checkpoint': config.hparams['LOAD_FROM_CHECKPOINT'],
                    'hparams/Learning_Rate': config.hparams['learning_rate'],
                    'transform/Transform': config.transform}
    
    return config_dict


def run_metrics(output_image, label_image, image_threshold) -> dict:
    iou = iou_metric(output_image, label_image, image_threshold)
    tn = true_negative(output_image, label_image, image_threshold)
    fn = false_negative(output_image, label_image, image_threshold)
    tp = true_positive(output_image, label_image, image_threshold)
    fp = false_positive(output_image, label_image, image_threshold)
    union_metric = union(output_image, label_image, image_threshold)
    jac = JAC(output_image, label_image, image_threshold)
    recall_metric = recall(output_image, label_image, image_threshold)
    specificity_metric = specificity(output_image, label_image, image_threshold)
    fallout_metric = fallout(output_image, label_image, image_threshold)
    fnr = FNR(output_image, label_image, image_threshold)
    ppv = PPV(output_image, label_image, image_threshold)

    metric_dict = {'test_metrics/IOU': iou,
                    'test_metrics/TN': tn,
                    'test_metrics/FN': fn,
                    'test_metrics/TP': tp,
                    'test_metrics/FP': fp,
                    'test_metrics/Union': union_metric,
                    'test_metrics/JAC': jac,
                    'test_metrics/Recall': recall_metric,
                    'test_metrics/Specificity': specificity_metric,
                    'test_metrics/Fallout': fallout_metric,
                    'test_metrics/FNR': fnr,
                    'test_metrics/PPV': ppv}
    
    return metric_dict


def iou_metric(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image> 0).type(torch.float32)
    intersection = torch.sum(torch.mul(output,label)).item()
    union = torch.sum(output + label > 0).item()
    total_iou = intersection/union
    return total_iou
    
def union(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image> 0 ).type(torch.float32)
    union = torch.sum(output + label > 0).item()
    total_union = union
    return total_union


def false_negative(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image> 0).type(torch.float32)
    false_neg = torch.sum(torch.sub(label, output) == 1).item()
    total_false_neg = false_neg
    return total_false_neg
    
    
def false_positive(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image> 0 ).type(torch.float32)
    false_pos =  torch.sum(torch.sub(output,label) == 1).item()
    total_false_pos = false_pos
    return total_false_pos


def true_positive(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0 ).type(torch.float32)
    true_pos = torch.sum(torch.mul(output,label) == 1).item()
    total_true_pos = true_pos
    return total_true_pos

def true_negative(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    true_neg = torch.sum(torch.mul(output,label) == 0).item()
    total_true_neg = true_neg
    return total_true_neg

def JAC(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    
    JAC = TP / (TP + FP + FN)
    total_JAC = JAC
    return total_JAC

def recall(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    
    recall = TP / (TP + FN)
    return recall

def specificity(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    
    spec = TN / (TN + FP)
    return spec
    
def fallout(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    
    fallout = FP / (FP + TN)
    return fallout
    
def FNR(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    
    FNR = FN / (FN + TP)
    return FNR

def PPV(output_image, label_image, threshold):
    output = (output_image > threshold).type(torch.float32)
    label = (label_image > 0).type(torch.float32)
    
    TP = torch.sum(torch.mul(output,label) == 1).item()
    FP = torch.sum(torch.sub(output,label) == 1).item()
    TN = torch.sum(torch.mul(output,label) == 0).item()
    FN = torch.sum(torch.sub(label,output) == 1).item()
    PPV = TP / (FP + TP + 1e-6)
    return PPV




# These are functions from the vanilla PyTorch JTML that are not currently used with Lightning

# not used because config was changed to .py module
def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/
    Args:
        log_path [str]: eg: "../log/train.log"
    """

    # parameter log_path specifies directory to use for logging
    # should be in the form ./log/[MODEL_NAME]/etl_[MODEL_NAME].log
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # configure the logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger


def load_data(processed_data):
    """
    Load data from specified file path
    ***I know this function is dumb, why we need another function? Just to demo unit test?
    In this case it is easy, but if you have complex pipeline, you will
    want to safeguard the behavior!
    Args:
        processed_data [str]: file path to processed data
    
    Returns:
        [tuple]: feature matrix and target variable
    """
    data = pd.read_csv(processed_data)
    return data.drop("quality", axis=1).to_numpy(), data["quality"], list(data.columns)

def load_neural_network(path_to_network, NN_model):
    '''
    The main reason for using this is to avoid having to worry about loading a NN in so many different places. 
    One of the main issues that this will overcome is the fact that for some nets, it is expecting "module" and others it is not.

    Args:
        path_to_network [str]: file path to the neural networks
        NN_model [torch.nn.Module]: object that contains the architecture to be loaded
        config [dataframe]: the configuration object used in all the different scripts
    Returns:
        network_state_dict [OrderedDict]: a dictionary with all the necessary terms
    '''

    # Create two objects with ordered dict for each of the different possibilities 
    with_mod = OrderedDict() 
    no_mod = OrderedDict()

    for old_name, w in torch.load(path_to_network, 
        map_location = {"cuda:0" : "cpu"})['model_state_dict'].items():
            if old_name[:7] == "module.":
                name_mod = old_name
                name_no_mod = old_name[7:]
            else:
                name_mod = "module." + old_name
                name_no_mod = old_name
            
            with_mod[name_mod] = w
            no_mod[name_no_mod] = w
        
    try:
        NN_model.load_state_dict(with_mod)
    except:
        NN_model.load_state_dict(no_mod)
    
    return "Sucessful"


def plot_print_test_predictions(test_image_batch,test_output_batch,img_name_batch,config,threshold,alpha):
    '''
    Values above the threshold will be categorized as a mask pixel, while values below will be categorized as
    background. Similar to the validation print function but uses every single image in batch and also
    saves the image in the order received. The alpha variable is the intensity with which we blend the red channel
    of the mask in the overlay picture. The start index is the index at which we begin numbering the saved images.
    '''
    rows = len(test_image_batch)
    # Plot image
    #fig =plt.figure(figsize=(16, 16*(rows/3)),dpi=180)
    #ax = []
    for img_index in range(0, len(test_image_batch)):
        I = test_image_batch[img_index].to("cpu").type(torch.float32)
        L = (test_output_batch[img_index].to("cpu") > threshold).type(torch.float32)
        img = torchvision.transforms.ToPILImage()(I.type(torch.uint8))
      # # ax.append( fig.add_subplot(rows, 3, 3*img_index + 1) )
      #  ax[-1].set_title("Image")
      #  plt.xticks([])
      #  plt.yticks([])
      #  plt.imshow(img)
        img = torchvision.transforms.ToPILImage()((255*L).type(torch.uint8))
        image_path = os.getcwd() + "/data/" + config.data_constants["MODEL_NAME"] +"/"+ config.data_constants["MODEL_TYPE"]+ "_" 
      # ax.append( fig.add_subplot(rows, 3, 3*img_index + 2) )
      # ax[-1].set_title("Prediction")
      # plt.xticks([])
      # plt.yticks([])
      # plt.imshow(img)
        
        img.save(image_path + 'test_prediction_'  + img_name_batch[img_index])
        img = torchvision.transforms.ToPILImage()(torch.cat([(1-alpha)*(L*I)+alpha*255*(L)+(1-L)*I,\
                                                                          (1-alpha)*(L*I)+(1-L)*I,\
                                                                          (1-alpha)*(L*I)+(1-L)*I],0)\
                                                   .type(torch.uint8))
        img.save(image_path + 'test_overlay_'+ img_name_batch[img_index])
      #  ax.append( fig.add_subplot(rows, 3, 3*img_index + 3))
      #  ax[-1].set_title("Overlay")
      #  plt.xticks([])
      #  plt.yticks([])
      #  plt.imshow(img)
    #plt.show()
