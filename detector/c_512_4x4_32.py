import numpy as np

# from config import Config
# from data import BALANCE_WEIGHTS
from .layers import *

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    # 'train_dir': 'data/train_medium',
    # 'test_dir': 'data/test_medium',
    # 'batch_size_train': 48,
    # 'batch_size_test': 8,
    # 'balance_weights': np.array(BALANCE_WEIGHTS),
    # 'balance_ratio': 0.975,
    # 'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    # 'aug_params': {
    #     'zoom_range': (1 / 1.15, 1.15),
    #     'rotation_range': (0, 360),
    #     'shear_range': (0, 0),
    #     'translation_range': (-40, 40),
    #     'do_flip': True,
    #     'allow_stretch': True,
    # },
    # 'sigma': 0.25,
    # 'schedule': {
    #     0: 0.003,
    #     150: 0.0003,
    #     220: 0.00003,
    #     251: 'stop',
    # },
}

def cp(filters, **kwargs):
    args = {
        'filters': filters,
        'kernel_size': (4, 4),
    }
    args.update(kwargs)
    return conv_params(**args)

n = 32

layers = [
    (InputLayer, {'input_shape': (cnf['h'], cnf['w'], 3)}),
    (Conv2D, cp(n, strides=(2, 2))),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(n,
                kernel_regularizer=l1_l2(regular_factor_l1, regular_factor_l2),
                bias_regularizer=l1_l2(regular_factor_l1, regular_factor_l2))),
    (MaxPool2D, pool_params()),
    (Conv2D, cp(2 * n, strides=(2, 2))),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(2 * n)),
    (Conv2D, cp(2 * n)),
    (MaxPool2D, pool_params()),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(4 * n)),
    (Conv2D, cp(4 * n)),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(4 * n)),
    (MaxPool2D, pool_params()),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(8 * n)),
    (Conv2D, cp(8 * n)),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(8 * n)),
    (MaxPool2D, pool_params()),
    (Conv2D, cp(16 * n)),
    (RMSPoolLayer, pool_params()),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}), (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), (MaxPooling1D, {'pool_size': 2}),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}), (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), (MaxPooling1D, {'pool_size': 2}),
    (Flatten, {}), (Dense, dense_params(1, activation='linear')),
]

# config = Config(layers=layers, cnf=cnf)
