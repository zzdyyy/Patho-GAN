"""prepare some layers and their default parameters for configs/*.py"""

import keras
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1_l2, l2

leaky_rectify_alpha = 0.01
regular_factor_l1 = 0.
regular_factor_l2 = 5e-4  # weight_decay

def conv_params(filters, **kwargs):
    """default Conv2d arguments"""
    args = {
        'filters': filters,
        'kernel_size': (3, 3),
        'padding': 'same',
        'activation': lambda x: keras.activations.relu(x, leaky_rectify_alpha),
        'use_bias': True,
        'kernel_initializer': 'zero',
        'bias_initializer': 'zero',
    }
    args.update(kwargs)
    return args


def pool_params(**kwargs):
    """default MaxPool2d/RMSPoolLayer arguments"""
    args = {
        'pool_size': 3,
        'strides': (2, 2),
    }
    args.update(kwargs)
    return args


def dense_params(num_units, **kwargs):
    """default dense layer arguments"""
    args = {
        'units': num_units,
        'activation': lambda x: keras.activations.relu(x, leaky_rectify_alpha),
        'kernel_initializer': 'zero',
        'bias_initializer': 'zero',
    }
    args.update(kwargs)
    return args


class RMSPoolLayer(keras.layers.pooling._Pooling2D):
    """Use RMS(Root Mean Squared) as pooling function.

        origin version from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
    """
    def __init__(self, *args, **kwargs):
        super(RMSPoolLayer, self).__init__(*args, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(K.square(inputs), pool_size, strides,
                          padding, data_format, pool_mode='avg')
        return K.sqrt(output + K.epsilon())
