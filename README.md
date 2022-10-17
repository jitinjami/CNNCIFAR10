## Image Classification Using ConvNets

This project is about implementing and training a Convolutional Neural Network to learn to distinguish between 10 different articles that are part of the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset contains 50 000 training images and 10 000 test images. Each image is of size 32×32 with three channels for red, green, and blue. Each image belongs to one of ten classes: plane, car, bird, cat, deer, dog, frog, horse, ship, or truck

## Motivation
This project was a part of Assignment 2 of the "Deep Learning Lab" course at USI, Lugano taken by [Dr. Kazuki Irie](https://people.idsia.ch/~kazuki/).

## Tech used
<b>Built with</b>
- [Python3](https://www.python.org)
- [NumPy](https://numpy.org)
- [PyTorch](https://pytorch.org)


## Features
The project includes implementation and exploration of the following ideas related to CNNs and Deep Learning projects in general:
- Normalizing the dataset within all 3 channels (red, green abd blue) using mean and standard deviation.
- Seeing the effect of regularization via dropout layers placed after Convolutional layers and Feed forward layer
- Hyperparameter tuning with the learning rates and dropout probabilities.

A report can be found explaining my findings in this repo titled `Report.pdf`

## Credits
Two articles helped me through the creation of this project:
- [Backpropagation through a fully-connected layer by Eli Bendersky](https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/) 
- [A simple neural net in numpy by Silvian Gugger](https://sgugger.github.io/a-simple-neural-net-in-numpy.html)