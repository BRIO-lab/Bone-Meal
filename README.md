# Lightning Segmentation

### This repo is for practicing using our workflow.



## Setup:

### Conda environment

1. Install Anaconda package manager with Python version 3.9 from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended because of small size) or [full Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (includes graphical user interface for package management).
2. Verify that the pip3 (Python 3's official package manager) is installed by entering `pip3 -v` in the terminal. If it is not installed, install it, perhaps using [this tutorial](https://www.activestate.com/resources/quick-reads/how-to-install-and-use-pip3/).
3. [Create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the conda environment `jtml` from the `environment.yml` using the command `conda env create -f environment.yml`.
4. Activate the conda env with `conda activate jtml`.
5. There may be other dependencies that you can install using conda or pip3.

### [WandB](https://wandb.ai/) - our logging system.

1. Create an account from the website and send the email you used to Sasank (to get invited to the Wandb team).

### CUDA (Optional)

If you have an NVIDIA graphics card, please install [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2136/~/how-to-install-cuda). This will allow you to use your GPU for training, which is useful when running a couple batches during development to ensure the code runs.

## Data:

Large data is in the Files section of the Microsoft Teams team. Please copy these files/folders locally. This includes the image folder of X-ray images and segmentation masks (you need to unzip this folder) and the .ckpt model checkpoint file needed for loading a pretrained model for testing.

After you download these files/folders locally, remember to edit the config file (in config/config.py) to specify the location of your local image directory and checkpoint file.

## Use:

1. Be in the LitJTML directory (use the `cd` command to change the directory to the `blah/blah/LitJTML/` directory).
2. To fit (train) a model, call `python scripts/fit.py my_config` where `my_config` is the name of the config.py file in the `config/` directory.
    - The config file should specify the model, data, and other parameters.
