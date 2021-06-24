# author is He Zhao
# The time to create is 8:49 PM, 28/11/16

import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.misc
import scipy
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def lrelu1(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.

def matTonpy():

    # img = sio.loadmat('sample.mat')['imgSample']
    # gt = sio.loadmat('sample.mat')['gtSample']

    img = sio.loadmat('test.mat')['imgAllTest']
    gt = sio.loadmat('test.mat')['gtAllTest']
		
    # with open('img_sample.npy', 'wb') as fout:
    #     np.save(fout, img)
    # with open('gt_sample.npy', 'wb') as fout:
    #     np.save(fout, gt)
    return img, gt


def TestImgForTest(dataPath):

    img = sio.loadmat(dataPath)['imgAllTest']
    gt = sio.loadmat(dataPath)['gtAllTest']

    return img, gt


def TrainImgForTest(dataPath):

    img = sio.loadmat(dataPath)['imgAllTrain']
    gt = sio.loadmat(dataPath)['gtAllTrain']

    return img, gt

def resUnit(input_layer, i, out_size):
    with tf.variable_scope("g_res_unit" + str(i)):
        net = slim.conv2d(inputs=input_layer, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                          num_outputs=out_size, kernel_size=[4, 4], stride=2, padding='SAME')

        net = slim.conv2d(inputs=net, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                          num_outputs=out_size, kernel_size=[4, 4], stride=1, padding='SAME')

        res = slim.conv2d(inputs=input_layer, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                          num_outputs=out_size, kernel_size=[1, 1], stride=2, padding='SAME')

        output = net + res
    return output


def resUnit_up(input_layer, i, out_size):
    with tf.variable_scope("g_res_unit_up" + str(i)):
        net = slim.conv2d_transpose(inputs=input_layer, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                    num_outputs=out_size, kernel_size=[4, 4], stride=1, padding='SAME')

        net = slim.conv2d_transpose(inputs=net, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                    num_outputs=out_size, kernel_size=[4, 4], stride=2, padding='SAME')

        res = slim.conv2d_transpose(inputs=input_layer, normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                    num_outputs=out_size, kernel_size=[1, 1], stride=2, padding='SAME')

        output = net + res
    return output
