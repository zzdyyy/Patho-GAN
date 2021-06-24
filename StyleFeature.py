# derived from He Zhao's version created at 3:40 PM, 23/3/17
import tensorflow as tf
import vgg
import detector
import numpy as np
import scipy.io as sio
from skimage import measure
from scipy import ndimage
import cv2


STYLE_LAYERS = ('conv2d_1', 'conv2d_3')
STYLE_LAYERS_SIZE = (256, 64)
STYLE_LAYERS_CHANNELS = (32, 64)
STYLE_LAYERS_MEAN = (2e-7, -2e-5)
STYLE_LAYERS_STD = (0.05, 0.03)
CONTENT_LAYER = ('relu4_2',)





class Lesion:
    """Descriptor of a single Lesion"""
    def __init__(self):
        self.sty_bbox = (0, 0, 0, 0)  # bbox(y1, y2, x1, x2) of the lesion in the style reference image
        self.inmask = None  # a patch of the input_mask that include all the lesion region
        self.feature = {}  # the feature map dictionary, {"layer_name": (bbox, activation, GRAM)}


def gauss_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.max(kernel)


def projection_to_mask(projection):
    with tf.name_scope('proj_to_fmask'):
        # projection = tf.reduce_sum(projection, axis=-1, keep_dims=True)  # fused across channels?
        projection = tf.abs(projection)
        projection = projection / (1e-20 + tf.reduce_max(projection))
        projection = tf.maximum(projection, 0.)
        # projection = tf.sqrt(projection)  # broaden?
        return projection


def get_input_mask(projection, dilation=2, threshold=0.33):
    with tf.name_scope('input_mask'):
        projection = tf.reduce_sum(projection, axis=-1, keepdims=True)  # fused across channels?
        projection = tf.abs(projection)
        projection = projection / (1e-20 + tf.reduce_max(projection))
        projection = tf.maximum(projection, 0.)
        projection = tf.to_float(projection > threshold)
        r = dilation
        x, y = np.ogrid[-r:r+1, -r:r+1]
        filter = (x**2 + y**2 <= r**2).astype(np.float32)[..., np.newaxis] - 1
        projection = tf.nn.dilation2d(projection, filter, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
        projection = projection
        return projection


def get_lesion_descriptors(sty_inmask, label_mask=None):
    """return a descriptor frameworks
    sty_inmask: mask to restrict style loss
    label_mask: used to find bbox and separate lesions
    """
    assert (sty_inmask.shape[0] == 448)
    assert (sty_inmask.shape[1] == 448)
    assert (label_mask.shape[0] == 448)
    assert (label_mask.shape[1] == 448)

    descriptors = []

    if label_mask is None:
        print('Use sty_inmask as label_mask!')
        label_mask = sty_inmask

    sty_inmask_labeled = measure.label(label_mask > 0, connectivity=2)  # find connected components
    assert sty_inmask_labeled.max() >= 1
    positions = ndimage.find_objects(sty_inmask_labeled)  # get bboxes

    for i in range(sty_inmask_labeled.max()):
        lesion = Lesion()
        slice1, slice2 = positions[i]
        lesion.sty_bbox = (slice1.start, slice1.stop, slice2.start, slice2.stop)
        lesion.inmask = ((sty_inmask[slice1, slice2] > 0) & (sty_inmask_labeled[slice1, slice2] == i+1)).astype(np.float32)
        descriptors.append(lesion)

    return descriptors


def fill_features_into_descriptors(descriptors, model, sess, feed_dict, activation_restriction=True):
    """use model to get style feature and put them in descriptors
    activation_restriction=True, use activation_map as a restriction
    activation_restriction=False, use simply in_mask to restrict
    """
    for i, layer in enumerate(STYLE_LAYERS):
        features = model.outputs[i]
        activation = projection_to_mask(model.output_projection[i])

        for j, lesion in enumerate(descriptors):  # type:Lesion
            y1, y2, x1, x2 = lesion.sty_bbox
            input_mask = np.pad(lesion.inmask, [[y1, 448-y2], [x1, 448-x2]], mode='constant')
            scaled_inmask = cv2.resize(input_mask, tuple(model.outputs[i].shape.as_list()[1:3]))
            slice1, slice2 = ndimage.find_objects(scaled_inmask > 0)[0]
            local_bbox = (slice1.start, slice1.stop, slice2.start, slice2.stop)
            local_inmask = scaled_inmask[None, slice1, slice2, None]
            local_activation = activation[:, slice1, slice2, :] * local_inmask \
                if activation_restriction else \
                (tf.ones([1, slice1.stop-slice1.start, slice2.stop-slice2.start, 1]) * local_inmask * 1.e-1)
            local_features = features[:, slice1, slice2, :] * local_activation
            local_features = tf.reshape(local_features, shape=[-1, local_features.shape.as_list()[1] * local_features.shape.as_list()[2],
                                                               local_features.shape.as_list()[3]])[0]
            local_features_T = tf.transpose(local_features)
            local_gram = tf.matmul(local_features_T, local_features) / float(local_features.shape.as_list()[0] * local_features.shape.as_list()[1])
            lesion.feature[layer] = (local_bbox,) + sess.run((local_activation, local_gram), feed_dict=feed_dict)


def get_style_model(image, mask, with_feature_mask_from=None):

    if mask is not None:
        image = (image+1)*((mask+1)/2)-1

    if image._shape_as_list()[1] != 448:
        image = tf.image.resize_images(image, [448,448])

    model = detector.get_layers_model(image, STYLE_LAYERS + ('dense_3',), 'style_model',
                                      with_projection_output_from=with_feature_mask_from)
    return model


def get_patho_loss(img_model, syn_model):
    return tf.reduce_mean(tf.square(
        img_model.get_layer('my_input').related_projection.output
        - syn_model.get_layer('my_input').related_projection.output
    ))  # MSE


def get_severity_loss(img_model, syn_model):
    return tf.reduce_mean(tf.square(
        img_model.get_layer('dense_3').output
        - syn_model.get_layer('dense_3').output
    ))  # MSE


def get_content_features(image, mask):
    image = tf.multiply(image + 1, 127.5)
    if mask is not None:
        image = image * ((mask + 1) / 2)

    img_features = {}

    if image._shape_as_list()[1] != 512:
        image = tf.image.resize_images(image, [512, 512])

    # with tf.device('/cpu:0'):

    img_pre = vgg.preprocess(image)
    vgg_path = 'data/imagenet-vgg-verydeep-19.mat'
    data = sio.loadmat(vgg_path)
    net = vgg.net(data, img_pre)

    for layer in CONTENT_LAYER:
        features = net[layer]
        img_features[layer] = features

    return img_features


def get_retinal_loss(img, syn, mask):

    img_features = get_content_features(img, mask)
    syn_features = get_content_features(syn, mask)

    content_lossE = 0
    for content_layer in CONTENT_LAYER:
        coff = float(1.0 / len(CONTENT_LAYER))
        img_content = img_features[content_layer]
        syn_content = syn_features[content_layer]
        content_lossE += coff * tf.reduce_mean(tf.abs(img_content - syn_content))

    content_loss = tf.reduce_mean(content_lossE)

    return content_loss


def get_tv_loss(img, mask, input_mask=None):
    # mask: [-1, 1]
    # input_mask: [0, 1]

    img = img*((mask+1)/2)
    # x = tf.reduce_sum(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
    # y = tf.reduce_sum(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    if input_mask is not None:
        x = tf.reduce_sum(input_mask[:, :-1, :, :] * tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :])) / (1e-8 + 3*tf.reduce_sum(input_mask))
        y = tf.reduce_sum(input_mask[:, :, :-1, :] * tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :])) / (1e-8 + 3*tf.reduce_sum(input_mask))
    else:
        x = tf.reduce_mean(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
        y = tf.reduce_mean(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    return x+y
    

