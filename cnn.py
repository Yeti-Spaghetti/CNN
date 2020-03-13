import numpy as np
np.set_printoptions(threshold=np.inf)

from matplotlib import pyplot as plt
%matplotlib inline

import scipy.signal
import math
import random
import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

class NeuralNetwork:
    def __init__(self, img):
        # self.convolve = [img] # replace with more appropriate name? e.g. input_
        self.convolve = []
        self.convolve.append(img)
        self.convolve = np.array(self.convolve)
        self.filterTuple = (1,3,3)
        self.filters = []
        self.filterBias = []
        self.maxPoolCount = 0
        self.lastLayer = None
        self.weights1 = None
        self.fcBias = 1

    def Conv2D(self, layers=32, convSize=3):
        self.maxPoolCount += 1
        self.ConvFilters(layers, convSize)

    def ConvFilters(self, layers=32, size=3):
        self.lastLayer = layers
        self.filterTuple = list(self.filterTuple)
        self.filterTuple[0] = layers
        self.filterTuple = tuple(self.filterTuple)

        self.filters.append((np.random.random_sample(self.filterTuple)))

    def activation_layer(self, activation='ReLu', derivative=False):
        for i in range(len(self.convolve)):
            if activation == 'ReLu':
                if derivative:
                    self.convolve[i] = 1*(self.convolve[i]>0)
                else:
                    self.convolve[i] = self.convolve[i]*(self.convolve[i]>0)
            if activation == 'Sigmoid':
                if derivative:
                    self.convolve[i] = 1/(1+np.ex(-self.convolve[i])) * (1-(1/(1+np.ex(-self.convolve[i]))))
                else:
                   self.convolve[i] = 1/(1+np.ex(-self.convolve[i]))
            if activation == 'LeakyReLu':
                pass
            
    def pad_columns(self, filtSize, img):
        # pad columns of zeros
        r = filtSize//2
        col = np.zeros((self.img.shape[0], r))
        paddedImg = np.hstack((self.img,col))
        paddedImg = np.hstack((col,paddedImg))  
        return paddedImg

    def pad_rows(self, filtSize, img):
        # pad row of zeros
        c = filtSize//2
        row = np.zeros((c, img.shape[1]))
        paddedImg = np.vstack((img, row))
        paddedImg = np.vstack((row, paddedImg))
        return paddedImg

    def MaxPool_padding(self, input_, poolSize, stride):
        if input_.ndim == 2:
            padding = True
            while (input_.shape[-1] - poolSize) % stride != 0:
                y,x = input_.shape
                # create temporary array
                temp = np.zeros((y, x+1))
                # create column of zeros
                colZeros = np.zeros((y, 1))
                if padding:
                    temp = np.c_[input_, colZeros]
                else:
                    temp = np.c_[colZeros, input_]
                padding = not padding
                input_ = temp

            while (input_.shape[-2] - poolSize) % stride != 0:
                input_T = input_.transpose
                y,x = input_T.shape
                #create temporary array
                temp = np.zeros((y, x+1))
                # create row of zeros
                colZeros = np.zeros((y, 1))
                if padding:
                    temp = np.c_[input_T, colZeros]
                else:
                    temp = np.c_[colZeros, input_T]
                padding = not padding
                input_ = temp.T
            return input_
        
        retArr = []
        for i in range(input_.shape[0]):
            retArr.append(self.MaxPool_padding(input_[i], poolSize, stride))
        retArr = np.array(retArr)
        return retArr

    # find maximum value in sliding window and return array
    def MaxPooling(self, input_, poolSize, stride):
        if input_.ndim == 2:
            rows = int(((input_.shape[0]-poolSize)/stride)+1)
            cols = int(((input_.shape[1]-poolSize)/stride)+1)
            maxPoolArr = np.zeros((rows, cols))

            for i in range(0, input_.shape[0]-stride, stride):
                for j in range(0, input_.shape[1]-stride, stride):
                    maxPoolArr[int(i/stride)][int(j/stride)] = input_[i:i+poolSize, j:j+poolSize].max()
            return maxPoolArr
        
        retArr = []
        for i in range(input_.shape[0]):
            retArr.append(self.MaxPooling(input_[i], poolSize, stride))
        retArr = np.array(retArr)
        return retArr

    def MaxPool(self, poolSize=2, stride=None):
        input_ = self.convolve

        if stride == None:
            stride = poolSize
        # add padding to image border if kernel doesn't divide evenly
        input_ = self.MaxPool_padding(input_, poolSize, stride)

        # perform max pool
        shape = input_.shape
        shapeList = list(shape)
        shapeList[-2] = int(((input_.shape[-2]-poolSize)/stride)+1)
        shapeList[-1] = int(((input_.shape[-1]-poolSize)/stride)+1)
        shape = tuple(shapeList)
        maxPoolArr = np.zeros(shape)

        # # find maximum value in sliding window and create new array
        self.convolve = self.MaxPooling(input_, poolSize, stride)

    def TrainCNN(self, epochs):
        self.filters = np.array(self.filters)
        self.weights1 = np.random.rand(self.lastLayer * int(self.convolve.shape[1] * self.convolve.shape[2] / 4**self.maxPoolCount))

        for e in range(epochs):
            # FORWARD PASS
            # convolution layer
            for i in range(len(self.filters)):
                temp = []
                for j in range(len(self.filters[i])):
                    convolveSum = np.zeros((self.convolve.shape[1], self.convolve.shape[2]))
                    for k in range(len(self.convolve)):
                        convolveSum += scipy.signal.correlate(self.convolve[k], self.filters[i][j], 'same')
                    temp.append(convolveSum)
                self.convolve = np.array(temp)

                # activation function layer
                self.activation_layer()

                # max-pool layer
                self.MaxPool()

            # fc layer
            fc = self.convolve.flatten()
            predict = np.dot(fc,self.weights1) + self.fcBias
            print(predict)

nn = NeuralNetwork(x_train[0])
nn.Conv2D(10)
nn.Conv2D(20)
nn.TrainCNN(1)
