"""create CNN model and train it"""

import tensorflow as tf
from . import c_512_4x4_32 as config
from .layers import *

_detector = None
models = {}


def get_detector():
    """return the detector singleton model, which is used for build layers model"""
    global _detector
    if _detector is not None:
        return _detector
    with tf.name_scope('detector_prototype'):
        _detector = keras.Sequential(name=config.cnf['name'])
        for layer, kwargs in config.layers:
            if 'activation' not in kwargs:
                _detector.add(layer(**kwargs))
                if layer is InputLayer:
                    _detector.add(Lambda(lambda x:x, name='my_input'))
            else:
                del kwargs['activation']
                new_layer = layer(**kwargs)
                new_layer.related_activation = LeakyReLU(leaky_rectify_alpha, name=new_layer.name+'_act')
                _detector.add(new_layer)
                _detector.add(new_layer.related_activation)
    return _detector


def get_layers_model(input, layer_names, scope_name='detector', load_weights_from='data/detector.h5', with_projection_output_from=None):
    """return a model, which output the features in determined layer_names
        argument:
        with_projection_output_from: (layer_name, channels), leave channels empty list to select all channels
    """
    proto = get_detector()
    with tf.name_scope(scope_name):
        model = keras.models.clone_model(proto, input)  # the model has his own weights and need to load_weights
        outputs = [
            model.get_layer(name).related_activation.output
                if hasattr(model.get_layer(name), 'related_activation')
                else model.get_layer(name).get_output_at(-1)
            for name in layer_names
        ]
        layers_model = keras.Model(input, outputs)
        models[scope_name] = layers_model

        if load_weights_from:
            model.load_weights(load_weights_from, by_name=True)

        layers_model.original_layers = model.layers
        if with_projection_output_from:
            layername, channels = with_projection_output_from
            _add_projection_network(layers_model, model.get_layer(layername), channels)
            layers_model.output_projection = [model.get_layer(name).related_projection.output for name in layer_names if hasattr(model.get_layer(name), 'related_projection')]

        return layers_model


def _add_projection_network(model, from_layer, channels):
    """add projection to model output, which is proposed originally for visualization in paper
       'Visualizing and Understanding Convolutio Networks'

       note that from_layer accepts a layer but [LeakyReLU, Dropout]
    """
    # get all layers to process on
    layers = [layer for layer in model.original_layers if not isinstance(layer, (LeakyReLU, Dropout))]

    def proj_func(input, layer: Layer):
        if isinstance(layer, (Dense, Conv2D)):
            input = tf.nn.relu(input)
            # input = input - layer.bias  # biased?
        output = tf.gradients(layer.output, layer.input, grad_ys=input)
        return output

    from_idx = layers.index(from_layer)
    to_idx = layers.index(model.get_layer('my_input'))
    x = from_layer.output
    if channels is not None and channels != []:
        new_layer = Lambda(lambda x: tf.gather(x, channels, axis=-1), name='gather_channel')
        x = new_layer(x)
        from_idx += 1
        layers[from_idx] = new_layer

    for idx in range(from_idx, to_idx, -1):
        new_layer = Lambda(proj_func, arguments={'layer': layers[idx]}, name=layers[idx-1].name+'_proj')
        x = new_layer(x)
        layers[idx-1].related_projection = new_layer


def preprocess(img):
    return tf.image.resize_images(img, [448, 448])
