
# COMP 6211H- Group 5 final project 

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

###  train/test
- Download a dataset (e.g. from https://doi.org/10.6084/m9.figshare.9250784.)

- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/your_own_dataset --name your_model_name --model cycle_gan 
```
To see more intermediate results, check out `./checkpoints/your_model_name/web/index.html`.


- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/your_own_dataset --name your_model_name --model cycle_gan
```
- The test results will be saved to a html file here: `./results/your_model_name/latest_test/index.html`.


## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.


## Acknowledgments
Our code is mainly inspired by [cycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and TransUNet(https://github.com/Beckschen/TransUNet).
