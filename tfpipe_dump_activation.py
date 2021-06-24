import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', help='a path with wildcard, remember to quote it in shell.')
parser.add_argument('--gpus')
parser.add_argument('--dump_to')
parser.add_argument('--visualize', action="store_true")
args = parser.parse_args()
print(args)

import cv2
import os
if args.gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])

import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
import detector
import glob

project_name = args.dump_to # for example 'IDRiD_train_activation_dump'
project_path = os.path.join('Visualization', project_name)
if not os.path.exists('Visualization'):
    os.makedirs('Visualization')

input_list = sorted(glob.glob(args.files))
print('total files:', len(input_list))

batch_size = 100
n_fill = -len(input_list) % batch_size  # fill n_fill elements to the last batch
input_list += [input_list[-1]] * n_fill

ds_fnames = tf.data.Dataset.from_tensor_slices(input_list)
# ds_images = ds_fnames.map(lambda fname: tf.image.decode_png(tf.read_file(fname), channels=3), num_parallel_calls=20)
ds_images = ds_fnames.map(lambda fname: tf.py_func(lambda fname: cv2.imread(fname.decode()), [fname], [tf.uint8], stateful=False), num_parallel_calls=24)
ds_images = ds_images.batch(batch_size)
ds_images = ds_images.prefetch(buffer_size=1)
images = ds_images.make_one_shot_iterator().get_next()
images = images[0]
images = images[..., ::-1]  # bgr to rgb

images = tf.cast(images, tf.float32)/255.0
images = (images - 0.5) * 2.0  # rescale to [-1, 1]
images.set_shape([None, 512, 512, 3])
images = tf.image.resize_images(images, [448, 448])


model = detector.get_layers_model(images, ['my_input'], 'scope_name',
                          with_projection_output_from=('dense_2', []))
model.summary()
projections = model.get_layer('my_input').related_projection.output
p_mean = tf.reduce_mean(projections, axis=[1, 2, 3], keepdims=True)
p_range = tf.reduce_max(projections, axis=[1, 2, 3], keepdims=True) - tf.reduce_min(projections, axis=[1, 2, 3], keepdims=True)
projections = (projections - p_mean) / 0.1 / 255 * 255.
projections = projections+0.5
projections = tf.clip_by_value(projections, 0., 1.)
projections = tf.image.resize_images(projections, [512, 512])

sess = K.get_session()

projection_list = []
try:
    i = 0
    while True:
        print(i*batch_size)
        projection_list.append(
            sess.run(projections, {K.learning_phase(): 0})  # K.learning_phase() 0:testing 1:training
        )
        i += 1
except tf.errors.OutOfRangeError:
    print('done.')
    pass

projection_npy = np.concatenate(projection_list, axis=0)
projection_npy = projection_npy[:len(projection_npy)-n_fill, ...]
print(len(projection_npy))
np.save(project_path+'.npy', projection_npy)
print('Result saved to', project_path+'.npy')

if args.visualize:
    if not os.path.exists(project_path):
        os.mkdir(project_path)

    for img, fname in zip(projection_npy, input_list):
        cv2.imwrite(
            os.path.join(project_path, os.path.basename(fname).replace('.jpg', '_A0.jpg')),
            img[..., ::-1]*255  # RGB2BGR
        )
        print('visualizing', os.path.join(project_path, os.path.basename(fname)), '...')