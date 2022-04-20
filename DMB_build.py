import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name')
parser.add_argument('--gpus')
args = parser.parse_args()
print(args)


import os
if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])

import keras.backend as K
import tensorflow as tf
import numpy as np
import StyleFeature
import cv2

from DMB_fragment import extract_descriptors
import pickle

# =============================== path set =============================================== #
dataset = args.dataset_name

img_channel = 3
img_size = 512
img_x = 512
img_y = 512
img = tf.placeholder(shape=[None, img_size, img_size, img_channel], dtype=tf.float32)
mask = tf.placeholder(shape=[None, img_size, img_size, 1], dtype=tf.float32)

img_detector = StyleFeature.get_style_model(img, mask, with_feature_mask_from=('dense_2', []))
act_input = {
    size: tf.image.resize_images((img_detector.get_layer(layer_name).related_projection.output - mean)/std, [size,size])
    for layer_name, size, mean, std in zip(StyleFeature.STYLE_LAYERS,
                                           StyleFeature.STYLE_LAYERS_SIZE,
                                           StyleFeature.STYLE_LAYERS_MEAN,
                                           StyleFeature.STYLE_LAYERS_STD)
}

img_sample = np.load('data/{}_test_image.npy'.format(dataset))  # [n, h, w, 3] original fundus
mask_sample = np.load('data/{}_test_mask.npy'.format(dataset))  # [n, h, w]    FOV mask
activation_maps = np.load('Visualization/{}_test.npy'.format(dataset))  # [n, h, w, 3] AMaps generated with tfpipe_dump_activation.py
segmentation_labels = np.load('data/{}_test_mask.npy'.format(dataset))[..., 1:]  # Deprecated

img_sample = (np.reshape(img_sample, [-1, img_x, img_y, img_channel]) - 0.5) * 2.0
mask_sample = (np.reshape(mask_sample, [-1, img_x, img_y, 1]) - 0.5) * 2.0
activation_maps = (activation_maps - 0.5) * 2.0


# extract all descriptors
descriptors = []
for i, (img_array, mask_array, amap_array, seg_label) in enumerate(zip(img_sample, mask_sample, activation_maps, segmentation_labels)):
    print('img:', i)

    intermed_amap = K.get_session().run(act_input, feed_dict={img: [img_array], mask: [mask_array]})

    descriptors.extend(extract_descriptors(intermed_amap, amap_array, seg_label, dataset, i))

if not os.path.exists('DMB'):
    os.makedirs('DMB')

# resort fragments by category
fragments_by_category = [[frag for frag in descriptors if frag[-2] == i] for i in range(len(img_sample))]
with open('DMB/{}.by_img'.format(dataset), 'wb') as file:
    pickle.dump(fragments_by_category, file)

# resort fragments by category
fragments_by_category = [[frag for frag in descriptors if frag[-1] == i] for i in range(-1, segmentation_labels.shape[-1])]
with open('DMB/{}.by_cat'.format(dataset), 'wb') as file:
    pickle.dump(fragments_by_category, file)