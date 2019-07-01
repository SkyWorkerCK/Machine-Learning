# Machine-Learning

![Binder](https://mybinder.org/badge_logo.svg)
![Build Status](https://travis-ci.org/trekhleb/homemade-machine-learning.svg?branch=master)

***

## MNIST-data

I adopt the MNIST data , as everyone knows , to train the model.  
> Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)

> Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)

> Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)

> Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)


## Full-Connected Neural Network(ANN)

Full-Connected Neural Network which consists of mnist_forward.py, mnist_backward.py, mnist_test.py and mnist_app.py. 
> mnist_forward.py

> mnist_backward.py

> mnist_test.py

> mnist_app.py


## Convolutional Neural Network(CNN)

Convolutional Neural Network which consists of mnist_lenet5_backward.py, mnist_lenet5_forward.py and mnist_lenet5_test.py. 
> mnist_lenet5_forward.py,

> mnist_lenet5_backward.py

> mnist_lenet5_test.py


## The folder "model"

The folder "model" is used to records the process which steps we are currently , in case that if the electricity cut off by accident, we can still check the model and keep runninng the process. Um..., it can save a lot of time as importantly.


## non-size_image and weird_number_image

These two folders contains digital-numbers from 0 to 9 which created by using PhotoShop . The digital-numbers in non-size_image are 28*28 dimension. All images are created as standard as possible while those in the folder , weird_number_image, are created as casual as possible to test the accuracy of the model .




