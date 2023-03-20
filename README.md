# Bone-Meal: Lightning Segmentation

### Segmentation and keypoint estimation for x-ray images of bones
### This repo is for practicing our workflow.


## More about Lightning Segmentation:
This project currently uses HRNet to perform segmentation on x-ray images of bones. However, better results are anticipated with HRFormer. The backbone of HRFormer is being added to the project to test and train with our data to yeild better results under some metrics like AP, AR, acc, #params, and FLOPs.
Torch.nn is a significant external class used in the realization of this project.
config.net will be the location of where models are defined and will have information defining a loss function. If the loss function is defined in config.net, the Backbone constructor will call loss_selector which has its call stack returning to fit.py. Otherwise, an error is returned.


## Setup:

### Conda environment

1. Install Anaconda package manager with Python version 3.9 from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended because of small size) or [full Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (includes graphical user interface for package management).
2. Verify that the pip3 (Python 3's official package manager) is installed by entering `pip3 -v` in the terminal. If it is not installed, install it, perhaps using [this tutorial](https://www.activestate.com/resources/quick-reads/how-to-install-and-use-pip3/).
3. [Create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the conda environment `jtml` from the `environment.yml` using the command `conda env create -f environment.yml`.
    Alternatively, use the win_modcuda_environment.yml or win_environment.yml to create the environment with windows.
4. Activate the conda env with `conda activate jtml`.
5. There may be other dependencies that you can install using conda or pip3.

### [WandB](https://wandb.ai/) - our logging system.

1. Create an account from the website and send the email you used to Sasank (to get invited to the Wandb team).
    - WandB is used because it effectively visualizes and logs our test and training runs of a model.
    - GPU Power, Memory, and Temp information is visualized. A train/loss graph is displayed with a validation output at specified step sizes. 

### CUDA (Optional)

If you have an NVIDIA graphics card, please install [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2136/~/how-to-install-cuda). This will allow you to use your GPU for training, which is useful when running a couple batches during development to ensure the code runs.

## Data:

Large data is in the Files section of the Microsoft Teams team. Please copy these files/folders locally. This includes the image folder of X-ray images and segmentation masks (you need to unzip this folder) and the .ckpt model checkpoint file needed for loading a pretrained model for testing.

After you download these files/folders locally, remember to edit the config file (in config/config.py) to specify the location of your local image directory and checkpoint file.

## Use:

1. Be in the Bone-Meal directory (use the `cd` command to change the directory to the `blah/blah/Bone-Meal/` directory).
    - Work within the anaconda prompt with your active environment.
2. To fit (train) a model, call `python scripts/fit.py my_config` where `my_config` is the name of the config.py file in the `config/` directory.
    - The config file should specify the model, data, and other parameters.
3. Initially, the project is set to run a small number of epochs. This can be modified in config.py
4. You will know know fitting your model worked if you view text that indicates a "success" for your run and the WandB logs function. WandB will indicate the state of running your model and whether it ran successfully or failed.
## Contribute:

- Source Code: https://github.com/BRIO-lab/Bone-Meal/
