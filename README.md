# Mix-DCGAN in Tensorflow

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage

First, download dataset with:

    $ python download.py mnist celebA

To train a model with downloaded dataset:

    $ python3 main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1 --is_train
    $ python main.py --dataset celebA --input_height=108 --is_train --is_crop True

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1
    $ python main.py --dataset celebA --input_height=108 --is_crop True

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --is_train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --c_dim=1 --is_train


## Comments

This Mix-DCGAN repository was build off Taehoon Kim's / [@carpedm20](http://carpedm20.github.io/) DCGAN implementation. The MIX-GAN is based off of the paper written in: [Generalization and Equilibrium in Generative Adversarial Nets ](https://arxiv.org/abs/1703.00573)


## Current Status
 - Weight Mixing with T=2 implemented
 - Current crashes with T > 2
 - Try with GPU machine?

## Author
Kenny Gea, Logan Engstrom 
