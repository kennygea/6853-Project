# Mix-DCGAN in Tensorflow

## Prerequisites

- Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage

First, download MNIST dataset with:

    $ python3 download.py mnist

To train a model with downloaded dataset:

    $ python3 main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1 --is_train

To test with an existing model:

    $ python3 main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1

## Comments

This Mix-DCGAN repository was build off Taehoon Kim's / [@carpedm20](http://carpedm20.github.io/) DCGAN implementation. The MIX-GAN is based off of the paper written in: [Generalization and Equilibrium in Generative Adversarial Nets ](https://arxiv.org/abs/1703.00573)

## Training and Testing with CelebA:
 - Download the image_align_celeba data set and put in in /data/[dir_name]
 - Then to train, run:

 	$ python3 main.py --dataset dir_name --input_height 64 --is_train --crop

## Current Status
 - Working with both MNIST and CelebA
 - Change self.T in model.py to change the number of Generator and Discriminators you want to train with

## Author
Kenny Gea, Logan Engstrom 
