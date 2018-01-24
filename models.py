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
        links = ChainList()

        for i_layer in range(n_layers):
            links.add_link(L.ConvolutionND(ndim=n_dim, in_channels=1 if i_layer==0 else n_filters, out_channels=n_filters,
                                           ksize=filter_size, stride=1, pad=pad_size,
                                           initialW=chainer.initializers.HeNormal()))
            links[-1].name = 'conv'
            if use_bn:
                links.add_link(L.BatchNormalization(n_filters))
                links[-1].name = 'bn'

        # links.add_link(L.Linear(None, 300))
        # links[-1].name = 'fc'
        links.add_link(L.Linear(None, n_output))
        links[-1].name = 'score'

        self.n_layers = len(links)
        self.h = {}
        super(CNN_ND, self).__init__(links)

    def __call__(self, x, t, test=False):

        for link in self[0][:-1]:
            x = link(x, test) if link.name == 'bn' else F.leaky_relu(link(x))

        self.y = self[0][-1](x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self

##
class MLP(ChainList):
    '''Multi-layered perceptron'''

    def __init__(self, n_layers, n_hidden, n_output, use_bn):

        # links = add_k_links(link_type='Linear', k=n_layers, n_in=None, n_hid=n_hidden, n_out=n_output)
        links = ChainList()
        for i_layer in range(n_layers-1):
            links.add_link(L.Linear(None, n_hidden))
            links[-1].name = 'fc'

            if use_bn:
                links.add_link(L.BatchNormalization(n_hidden))
                links[-1].name = 'bn'

        links.add_link(L.Linear(None, n_output))
        links[-1].name = 'score'
        self.h = {}
        self.n_layers = n_layers
        super(MLP, self).__init__(links)

    def __call__(self, x, t, test=False):
        for link in self[0][:-1]:
            x = link(x, test) if link.name == 'bn' else F.leaky_relu(link(x))

        self.y = self[0][-1](x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        return self


##
# class RNN(ChainList):
#     '''For ElmanRNN, LSTM, StatefulGRU'''
#     def __init__(self, n_layers, n_input, n_hidden, n_output, link_type='LSTM'):
#
#         links = add_k_links(link_type=link_type, k=n_layers, n_in=n_input, n_hid=n_hidden, n_out=n_output)
#
#         self.hidden = {}
#         self.n_layers = n_layers
#         super(RNN, self).__init__(links)
#
#     def __call__(self, x, t):
#
#         if self.n_layers > 1:
#             self.hidden[0] = self[0][0](x)
#             for i_layer in range(1, self.n_layers - 1):
#                 self.hidden[i_layer] = self[0][ilayer](self.hidden[i_layer - 1])
#             self.y = self[0][-1](self.hidden[self.n_layers - 2])
#         else:
#             self.y = self[0][0](x)
#
#         self.loss = F.mean_squared_error(t, self.y)
#         return self
#
#
# class MemoryRNN(RNN):
#     def __init__(self, n_layers, n_input, n_hidden, n_output, link_type='LSTM'):
#         RNN.__init__(self, n_layers, n_input, n_hidden, n_output, link_type)
#
#     def reset_state(self):
#         for i_layer in range(self.n_layers - 1):
#             self[0][i_layer].reset_state()
#
#
# class CNN_LSTM(Chain):
#     def __init__(self, n_input, n_filters, n_hidden, n_output, filter_size, pad_size):
#         super(CNN_LSTM, self).__init__(
#             l1=L.ConvolutionND(ndim=1, in_channels=n_input, out_channels=n_filters,
#                                ksize=filter_size, stride=1, pad=pad_size,
#                                initialW=chainer.initializers.HeNormal()),
#             l2=L.LSTM(n_filters, n_hidden),
#             l3=L.Linear(n_hidden, n_output)
#         )
#
#     def __call__(self, x, t, use_bn=False):
#         if len(x.shape) < 3: x = np.expand_dims(x, axis=2)
#         self.h1 = F.relu(F.batch_normalization(self.l1(x))) if use_bn else F.relu(self.l1(x))
#         self.h2 = self.l2(self.h1)
#         self.y  = self.l3(self.h2)
#
#         self.loss = F.mean_squared_error(t, self.y)
#
#         return self
#
#
# ##
# def add_k_links(link_type, k, n_in, n_hid, n_out):
#     links = ChainList()
#     if k > 1:
#         links.add_link(generate_link(link_type, n_in, n_hid))
#         for ilayer in range(2, k):
#             links.add_link(generate_link(link_type, n_hid, n_hid))
#         links.add_link(L.Linear(n_hid, n_out))
#     else:
#         links.add_link(L.Linear(n_in, n_out))
#
#     return links
#
# ##
# def generate_link(link_type, n_in, n_out):
#     return getattr(L, link_type)(n_in, n_out)