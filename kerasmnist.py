#from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1671) #for reproducibility

#network and training
NB_epoch = 200      #number of iterations for training
batch_size = 128    #batches for gradient descent
verbose = 1
NB_classes = 10     #number of outputs: Y = [0, ..., 9]
optimizer = SGD()
n_hidden = 128      #number of hidden layers
validation = .2     #% of train reserved for validation

#data needs to be shuffled and split between TRAIN and TEST sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()        #from the mnist set, load_data returns 2 set of tuples
#X_train is 60000 x 28 x 28 (an array of arrays)
#need to unravel the pixels into 60000 x 784
reshaped = 784
X_train = X_train.reshape(60000, reshaped)      #giving a new shape to numpy array without changing data
X_test = X_test.reshape(10000, reshaped)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

#convert vectors into binary class matrices since Y is a multiclass classification vector
Y_train = np_utils.to_categorical(y_train, NB_classes)
Y_test = np_utils.to_categorical(y_test, NB_classes)
