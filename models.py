import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np


##

class CNN_ND(ChainList):
    '''ND Convolutional neural network'''
    def __init__(self, n_layers, n_dim, n_filters, filter_size, pad_size, n_output, use_bn):
        links = []

        for i_layer in range(n_layers):
            links.append(L.ConvolutionND(ndim=n_dim, in_channels=1 if i_layer==0 else n_filters, out_channels=n_filters,
                                           ksize=filter_size, stride=1, pad=pad_size,
                                           initialW=chainer.initializers.HeNormal()))
            if use_bn:
                links.append(L.BatchNormalization(n_filters))

        links.append(L.Linear(None, n_output))

        self.n_layers = len(links)
        self.h = {}
        super(CNN_ND, self).__init__(*links)

    def __call__(self, x, t, test=False):

        for link in self[:-1]:
            x = link(x, test) if hasattr(link, 'avg_mean') else F.leaky_relu(link(x))

        self.y = self[-1](x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self


##
class MLP(ChainList):
    '''Multi-layered perceptron'''

    def __init__(self, n_layers, n_hidden, n_output, use_bn):

        links = []
        for i_layer in range(n_layers-1):
            links.append(L.Linear(None, n_hidden))

            if use_bn:
                links.append(L.BatchNormalization(n_hidden))

        links.append(L.Linear(None, n_output))
        self.h = {}
        self.n_layers = n_layers
        super(MLP, self).__init__(*links)

    def __call__(self, x, t, test=False):
        for link in self[:-1]:
            x = link(x, test) if hasattr(link, 'avg_mean')  else F.leaky_relu(link(x))

        self.y = self[-1](x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self

##
class Logistic(ChainList):
    def __init__(self, n_classes):
        links = []
        links.append(L.Linear(None, n_classes))
        super(Logistic, self).__init__(*links)

    def __call__(self, x, t, test=False):
        self.y = self(x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self


class CNN_ND_GAP(ChainList):
    '''ND Convolutional neural network with GAP'''
    def __init__(self, n_output):
        links = []

        links.append(L.ConvolutionND(2, 1, 32, 3, 1, 1, initialW=chainer.initializers.HeNormal()))
        links.append(L.ConvolutionND(2, 32, 32, 3, 1, 1, initialW=chainer.initializers.HeNormal()))
        links.append(L.BatchNormalization(32))

        links.append(L.ConvolutionND(2, 32, 64, 3, 1, 1, initialW=chainer.initializers.HeNormal()))
        links.append(L.ConvolutionND(2, 64, 64, 3, 1, 1, initialW=chainer.initializers.HeNormal()))
        links.append(L.BatchNormalization(64))

        links.append(L.Linear(None, n_output))

        self.n_layers = len(links)
        self.h = {}
        super(CNN_ND_GAP, self).__init__(*links)

    def __call__(self, x, t, test=False):

        for link in self[:-1]:
            x = link(x, test) if hasattr(link, 'avg_mean') else F.relu(link(x))
            if hasattr(link, 'avg_mean'): x = F.max_pooling_2d(x, ksize=3, stride=2, pad=1)

        # z = F.average_pooling_2d(x, ksize=x.shape[2:], stride=1)
        print(x.shape)
        # print(z.shape)
        # z = F.reshape(z, x.shape[:2])
        # self.y = self[0][-1](z)
        # print(z.shape)
        self.y = self[-1](x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self