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
        self.convActivation = []

        self.filterTuple = (1,3,3)
        self.filters = []
        self.maxPoolCount = 0

        # self.lastLayer = None
        # self.weights1 = None

        self.fcLayersSizes = []
        self.fcLayers = []
        self.fcLayersActivations = []

    def Conv2D(self, filters=32, convSize=3, activation='relu'):
        self.maxPoolCount += 1
        self.convActivation.append(activation)
        self.ConvFilters(filters, convSize)

    def ConvFilters(self, filters=32, size=3):
        self.lastLayer = filters
        self.filterTuple = list(self.filterTuple)
        self.filterTuple[0] = filters
        self.filterTuple = tuple(self.filterTuple)

        self.filters.append(np.random.uniform(-0.1,0.1,self.filterTuple))

    def activation(self, activation='ReLu', derivative=False):
        activation = activation.lower()
        for i in range(len(self.convolve)):
            if activation == 'relu':
                if derivative:
                    self.convolve[i] = 1*(self.convolve[i]>0)
                else:
                    self.convolve[i] = self.convolve[i]*(self.convolve[i]>0)

            if activation == 'sigmoid':
                if derivative:
                    self.convolve[i] = 1/(1+np.ex(-self.convolve[i])) * (1-(1/(1+np.ex(-self.convolve[i]))))
                else:
                   self.convolve[i] = 1/(1+np.ex(-self.convolve[i]))

            if activation == 'leakyreLu':
                self.convolve[i] = np.where(self.convolve > 0, self.convolve, self.convolve * 0.01)

            if activation == 'softmax':
                expSum = np.sum(np.exp(self.convolve))
                for i in range(len(self.convolve)):
                    self.convolve[i] = np.exp(self.convolve[i])/expSum

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
                input_T = np.transpose(input_)
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
    
    def MaxPool_padding_helper(self, input_, poolSize, stride):
        pass

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

    def fcLayer(self, size=4096, activation='relu'):
        self.fcLayersSizes.append(size)
        self.fcLayersActivations.append(activation)

    def fcLayersCreate(self):
        for i in range(len(self.fcLayersSizes)-1):
            fcLayer = np.random.uniform(-0.1,0.1,(self.fcLayersSizes[i], self.fcLayersSizes[i+1]))
            self.fcLayers.append(fcLayer)
        self.fcLayers = np.array(self.fcLayers)
 
    def fcLayersTrain(self):
        for i in range(len(self.fcLayers)):           
            self.convolve = self.convolve.dot(self.fcLayers[i])
            self.activation(self.fcLayersActivations[i])
            
    def TrainCNN(self, epochs):
        # necessary conversion for numpy operations
        self.filters = np.array(self.filters)
        # calculate size of flattened convolutional layer
        self.fcLayersSizes.insert(0,self.lastLayer * math.ceil(self.convolve.shape[1]/2**self.maxPoolCount) * math.ceil(self.convolve.shape[2]/2**self.maxPoolCount))
        # initialize fc layers
        self.fcLayersCreate()

        for e in range(epochs):
            # FORWARD PASS
            # convolution layer
            for i in range(0,len(self.filters)):
                temp = []
                for j in range(len(self.filters[i])):
                    convolveSum = np.zeros((self.convolve.shape[1], self.convolve.shape[2]))
                    for k in range(len(self.convolve)):
                        convolveSum += scipy.signal.correlate(self.convolve[k], self.filters[i][j], 'same')
                    temp.append(convolveSum)
                self.convolve = np.array(temp)

                # activation function layer
                self.activation(self.convActivation[i])

                # max-pool layer
                self.MaxPool()

            # flatten conv2d output
            self.convolve = self.convolve.flatten()
            self.convolve = self.convolve.reshape((1,self.convolve.shape[0]))
            # fully-connected layer step
            self.fcLayersTrain()

            # BACKPROPAGATION


nn = NeuralNetwork(x_train[0])
nn.Conv2D(10, activation='relu')
nn.Conv2D(20, activation='relu')
nn.Conv2D(30, activation='relu')
nn.fcLayer(4096, 'relu')
nn.fcLayer(4096, 'relu')
nn.fcLayer(10, 'SoftMax')

nn.TrainCNN(epochs=1)
