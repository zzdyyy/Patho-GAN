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
import Net
import numpy as np
import StyleFeature
import scipy.io as sio
import cv2
from DMB_fragment import rebuild_AMaps_by_img
import pickle
import random
import yaml

from Opts import save_images, matTonpy
# =============================== path set =============================================== #
out_dir = args.dataset_name + '_Reconstruct'
load_model = args.dataset_name
db_dataset = args.dataset_name
real_img_dataset = args.dataset_name # 'DRIVE'
real_img_test_dataset = args.dataset_name # 'DRIVE'

result_dir = 'Test' + '/' + out_dir + ''
model_directory = 'Model_and_Result' + '/' + load_model + '/models'  # Directory to restore trained model from.

if tf.gfile.Exists(result_dir):
    print('Result dir exists! Press Enter to OVERRIDE...', end='')
    input()
    tf.gfile.DeleteRecursively(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

os.system('cp {} {}'.format(__file__, result_dir))

# ============================== parameters set ========================================== #
max_epoch = 1


img_channel = 3
img_size = 512
img_x = 512
img_y = 512
gt_channel = 1

z_size = 400



# =============================== model and data definition ================================ #
generator = Net.generator

tf.reset_default_graph()

gt = tf.placeholder(shape=[None, img_size, img_size, gt_channel], dtype=tf.float32)
img = tf.placeholder(shape=[None, img_size, img_size, img_channel], dtype=tf.float32)
mask = tf.placeholder(shape=[None, img_size, img_size, 1], dtype=tf.float32)
z = tf.placeholder(shape=[None, z_size], dtype=tf.float32)

# gt_mask = tf.concat([gt, mask], 3)

img_detector = StyleFeature.get_style_model(img, mask, with_feature_mask_from=('dense_2', []))
act_input = {
    size: tf.image.resize_images((img_detector.get_layer(layer_name).related_projection.output - mean)/std, [size,size])
    for layer_name, size, mean, std in zip(StyleFeature.STYLE_LAYERS,
                                           StyleFeature.STYLE_LAYERS_SIZE,
                                           StyleFeature.STYLE_LAYERS_MEAN,
                                           StyleFeature.STYLE_LAYERS_STD)
}

syn = generator(gt, act_input, z)


# =============================== init ============================================= #
t_vars = tf.trainable_variables()
g_vars = list(set(t_vars) & set(tf.global_variables('generator')))

init = tf.variables_initializer(g_vars)
sess = K.get_session()
saver = tf.train.Saver(g_vars, max_to_keep=None)

sess.run(init)
#writer = tf.train.SummaryWriter(model_directory, sess.graph)

# ==================================== restore weights ================================ #
ckpt = tf.train.get_checkpoint_state(model_directory)
restore_path = ckpt.model_checkpoint_path.replace('-9000', '-9000')
print(restore_path)
saver.restore(sess, restore_path)



# ==================================== start training ===================================== #

if real_img_test_dataset in ['FGADR', 'retinal-lesions', 'IDRiD']:
    # img_sample = np.load('data/{}_test_image.npy'.format(real_img_test_dataset))
    gt_sample = np.load('data/{}_test_gt.npy'.format(real_img_test_dataset))[..., [0]]
    mask_sample = np.load('data/{}_test_mask.npy'.format(real_img_test_dataset))
elif real_img_test_dataset == 'DRIVE':
    img_sample, gt_sample, mask_sample = Net.matTonpy_35()


# img_sample = (np.reshape(img_sample, [-1, img_x, img_y, img_channel]) - 0.5) * 2.0
gt_sample = (np.reshape(gt_sample, [-1, img_x, img_y, gt_channel]) - 0.5) * 2.0
mask_sample = (np.reshape(mask_sample, [-1, img_x, img_y, 1]) - 0.5) * 2.0

per_part = 250
part_id = -1

with open('data/'+real_img_test_dataset+'_test.list', 'r') as f:
    fname_list = yaml.safe_load(f)

# amaps = rebuild_AMaps_by_img(0, fragments_DB)

# generate images with fragmentDB
for epoch in range(max_epoch):
    print('epoch:', epoch)
    batchNum = 1

    for i, (gt_array, mask_array) in enumerate(zip(gt_sample, mask_sample)):
        print('img:', batchNum)

        if i//per_part != part_id:
            part_id = i//per_part
            with open('DMB/{}.by_img.{}'.format(db_dataset, part_id), 'rb') as file:
                fragments_DB = pickle.load(file)

        zs = np.random.normal(0, 0.001, size=[1, z_size]).astype(np.float32)

        amaps = rebuild_AMaps_by_img(i % per_part, fragments_DB)
        syn_array = sess.run(syn, feed_dict={gt: [gt_array], z:zs, mask: [mask_array],
                                             act_input[256]: [amaps[256]],
                                             act_input[64]: [amaps[64]]})
        syn_array = (syn_array + 1) / 2
        syn_sample_m = syn_array * ((mask_array + 1) / 2)

        syn_sample_m = np.reshape(syn_sample_m, [img_x, img_y, img_channel])
        save_images(np.reshape(syn_sample_m, [1, img_x, img_y, img_channel]),
                    [1, 1],
                    result_dir + '/{}_{}.jpg'.format(fname_list[i], epoch))

        batchNum += 1

sess.close()
