# author is He Zhao
# The time to create is 8:47 PM, 28/11/16

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Opts import lrelu, resUnit
from dataBlocks import DataBlocks
import pickle
import scipy.io as sio
import numpy as np


initializer = tf.truncated_normal_initializer(stddev=0.02)
bias_initializer = tf.constant_initializer(0.0)


def build_data(batchsize, dataset_name='DRIVE'):
    #readpath = open('img.pkl', 'rb')
    #datapaths = pickle.load(readpath)
    datapaths = [['data/{}_train_image.npy'.format(dataset_name),
                  'data/{}_train_gt.npy'.format(dataset_name),
                  'data/{}_train_mask.npy'.format(dataset_name)]]
    db = DataBlocks(data_paths=datapaths, train_valid_ratio=[39, 0], batchsize=batchsize, allow_preload=False)
    return db
   


def matTonpy_35():

    img = sio.loadmat('test_1To4.mat')['imgAllTest']
    gt = sio.loadmat('test_1To4.mat')['gtAllTest']
    mask = sio.loadmat('test_1To4.mat')['maskAllTest']

    return img, gt, mask

def discriminator(image, reuse=False):
    n=32
    bn = slim.batch_norm
    with tf.variable_scope("discriminator"):
        # original
        dis1 = slim.convolution2d(image, n, [4, 4], 2, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer) # 256 256 64

        dis2 = slim.convolution2d(dis1, 2*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)  # 128 128 64

        dis3 = slim.convolution2d(dis2, 4*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)  # 64 64 128

        dis4 = slim.convolution2d(dis3, 8*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv4', weights_initializer=initializer)  # 32 32 256

        dis5 = slim.convolution2d(dis4, 16*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv5', weights_initializer=initializer)  # 16 16 512

        
        d_out_logits = slim.fully_connected(slim.flatten(dis5), 1, activation_fn=None, reuse=reuse, scope='d_out',
                                            weights_initializer=initializer)

        d_out = tf.nn.sigmoid(d_out_logits)
    return d_out, d_out_logits
    
def generator(image, act_input, z):
    n = 64
    with tf.variable_scope("generator"):
        # original
        e1 = slim.conv2d(image, n, [4, 4], 2, activation_fn=lrelu, scope='g_e1_conv',
                         weights_initializer=initializer)
        # 256
        e2 = slim.conv2d(tf.concat([lrelu(e1), act_input[256]], 3), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e2_conv',
                         weights_initializer=initializer)
        # 128
        e3 = slim.conv2d(lrelu(e2), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e3_conv',
                         weights_initializer=initializer)
        # 64
        e4 = slim.conv2d(tf.concat([lrelu(e3), act_input[64]], 3), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e4_conv',
                         weights_initializer=initializer)
        # 32
        e5 = slim.conv2d(lrelu(e4), 8*n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e5_conv',
                         weights_initializer=initializer)
        # 16
        e6 = slim.conv2d(lrelu(e5), 8*n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e6_conv',
                         weights_initializer=initializer)
        # 8


        zP = slim.fully_connected(z, 4 * 4 * n, normalizer_fn=None, activation_fn=lrelu, scope='g_project',
                                  weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 4, 4, n])

        gen1 = slim.conv2d(tf.image.resize_nearest_neighbor(lrelu(zCon), [8, 8]),
                           2 * n, [3, 3], 1, padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv1', weights_initializer=initializer)
        # 8
        gen1 = tf.concat([gen1, e6], 3)
        
        gen2 = slim.conv2d(tf.image.resize_nearest_neighbor(lrelu(gen1), [16, 16]),
                           4 * n, [3, 3], 1, normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv2', weights_initializer=initializer)
        # 16
        gen2 = tf.concat([gen2, e5], 3)

        gen3 = slim.conv2d(tf.image.resize_nearest_neighbor(lrelu(gen2), [32, 32]),
                           8 * n, [3, 3], 1, normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv3', weights_initializer=initializer)
        gen3 = tf.concat([gen3, e4], 3)

        # 32
        gen6 = slim.conv2d(tf.image.resize_nearest_neighbor(tf.nn.relu(gen3), [64, 64]),
                           4 * n, [3, 3], 1, normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv6', weights_initializer=initializer)
        gen6 = tf.concat([gen6, e3], 3)

        # 64
        gen7 = slim.conv2d(tf.image.resize_nearest_neighbor(tf.nn.relu(gen6), [128, 128]),
                           2 * n, [3, 3], 1, normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv7', weights_initializer=initializer)
        gen7 = tf.concat([gen7, e2], 3)

        # 128
        gen8 = slim.conv2d(tf.image.resize_nearest_neighbor(tf.nn.relu(gen7), [256, 256]),
                           n, [3, 3], 1, normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='g_dconv8', weights_initializer=initializer)
        # 256
        # gen8 = tf.nn.dropout(gen8, 0.5)
        gen8 = tf.concat([gen8, e1], 3)
        gen8 = tf.nn.relu(gen8)

        # 256
        gen_out = slim.conv2d(tf.image.resize_nearest_neighbor(gen8, [512, 512]),
                              3, [3, 3], 1, activation_fn=tf.nn.tanh, scope='g_out',
                              weights_initializer=initializer)

    return gen_out    
    
