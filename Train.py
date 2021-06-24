# derived from He Zhao's version created at 10:23 AM, 29/11/16
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name')
parser.add_argument('test1')
parser.add_argument('test2')
parser.add_argument('test3')
parser.add_argument('test4')
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
import detector
from tensorflow.python.training import training_util
import Net
import numpy as np
import os
import time
import StyleFeature
import scipy.io as sio
from scipy import ndimage
import imageio
import utils
import pdb
import cv2
import threading
import queue
#import matplotlib.pyplot as plt
import pickle
import yaml

from Opts import save_images, matTonpy
# =============================== path set =============================================== #
load_model = None # 'pe_nosty_base'  # 'initial_model'
save_model = False
real_img_dataset = args.dataset_name  #'retinal-lesions' # 'DRIVE'
real_img_test_dataset = args.dataset_name  #'retinal-lesions' # 'DRIVE'

# ============================== parameters set ========================================== #
#adversarial
w_adv = 1
#pathological
w_patho = 1e5  # 50e6
#retinal_details
w_retinal = 1
#tv
w_tv = 100
#severity
w_severity = 10


# ============================== model set ========================================== #
model_name = args.dataset_name

result_dir = 'Model_and_Result' + '/' + model_name + ''
sample_directory = result_dir + '/figs'
sample_directory2 = result_dir + '/figs_mask'
summary_dir = result_dir + '/summary'
# Directory to save sample images from generator in.
model_directory = result_dir + '/models'  # Directory to save trained model to.

print('Experiment name:', model_name)

if tf.gfile.Exists(result_dir):
    print('Result dir exists! Press Enter to OVERRIDE...', end='')
    input()
    tf.gfile.DeleteRecursively(result_dir)
if not os.path.exists(sample_directory):
    os.makedirs(sample_directory)
if not os.path.exists(sample_directory2):
    os.makedirs(sample_directory2)
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

os.system('cp {} {}'.format(__file__, result_dir))
os.system('cp {} {}'.format('Net.py', result_dir))
os.system('cp {} {}'.format('Opts.py', result_dir))
os.system('cp {} {}'.format('StyleFeature.py', result_dir))

with open(model_directory + '/training_log.txt', 'w') as f:
    f.close()
# ============================== parameters set ========================================== #

learning_rate = 0.0002 / 5
beta1 = 0.5

batch_size = 1  # Size of image batch to apply at each iteration.
max_epoch = 1000

img_channel = 3
img_size = 512
img_x = 512
img_y = 512
padding_l = 0
padding_r = 0
padding_t = 0
padding_d = 0
gt_channel = 1

style_size = 512

sample_batch = 4
z_size = 400



# =============================== model and data definition ================================ #
from Net import generator, discriminator, build_data

tf.reset_default_graph()

gt = tf.placeholder(shape=[None, img_size, img_size, gt_channel], dtype=tf.float32)
img = tf.placeholder(shape=[None, img_size, img_size, img_channel], dtype=tf.float32)
mask = tf.placeholder(shape=[None, img_size, img_size, 1], dtype=tf.float32)
z = tf.placeholder(shape=[None, z_size], dtype=tf.float32)

# gt_mask = tf.concat([gt, mask], 3)
# gt_mask = tf.Print(gt_mask, [tf.reduce_any(tf.is_nan(gt_mask))], 'is_nan(gt_mask):')
# zz = tf.Print(z, [tf.reduce_any(tf.is_nan(z))], 'is_nan(z):')

img_detector = StyleFeature.get_style_model(img, mask, with_feature_mask_from=('dense_2', []))
act_input = {
    size: tf.image.resize_images((img_detector.get_layer(layer_name).related_projection.output - mean)/std, [size,size])
    for layer_name, size, mean, std in zip(StyleFeature.STYLE_LAYERS,
                                           StyleFeature.STYLE_LAYERS_SIZE,
                                           StyleFeature.STYLE_LAYERS_MEAN,
                                           StyleFeature.STYLE_LAYERS_STD)
}

projection = (img_detector.get_layer('my_input').related_projection.output - 1e-6) / 0.05
projection = tf.clip_by_value(projection, -0.5, 0.5)
projection = projection * 2
projection = tf.abs(projection)
projection = tf.reduce_mean(projection, 3, keepdims=True)
projection = tf.nn.conv2d(projection, StyleFeature.gauss_kernel(31, 10)[..., None, None], [1,1,1,1],'SAME')  # gauss blur
projection = projection / (tf.reduce_max(projection) + 1e-7)
binary_mask = tf.py_func(lambda gray: cv2.threshold((gray[0, ..., 0]*255).astype('uint8'), 0, 255, cv2.THRESH_OTSU)[1][None, ..., None],
                         [projection], tf.uint8)
binary_mask256 = tf.cast(tf.image.resize_bilinear(binary_mask, [256, 256]) > 0, 'float32')
binary_mask64 = tf.cast(tf.image.resize_bilinear(binary_mask, [64, 64]) > 0, 'float32')

masked_act_input = {
    256: binary_mask256 * act_input[256],
    64: binary_mask64 * act_input[64],
}

syn = generator(gt, masked_act_input, z)
# syn = tf.Print(syn, [tf.reduce_any(tf.is_nan(syn))], 'is_nan(syn):')

real_img_gt = tf.concat([img*((mask+1)/2), gt], 3)
fake_syn_gt = tf.concat([syn*((mask+1)/2), gt], 3)
# real_img_gt = tf.Print(real_img_gt, [tf.reduce_any(tf.is_nan(real_img_gt))], 'is_nan(real_img_gt):')
# fake_syn_gt = tf.Print(fake_syn_gt, [tf.reduce_any(tf.is_nan(fake_syn_gt))], 'is_nan(fake_syn_gt):')

Dx, Dx_logits = discriminator(real_img_gt)
# Dx = tf.Print(Dx, [tf.reduce_any(tf.is_nan(Dx))], 'is_nan(Dx):')
Dg, Dg_logits = discriminator(fake_syn_gt, reuse=True)

db = build_data(batch_size, dataset_name=real_img_dataset)

syn_detector = StyleFeature.get_style_model(syn, mask, with_feature_mask_from=('dense_2', []))

# ============================================================================================#
# discriminator loss
with tf.name_scope('d_loss'):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logits, labels=tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logits, labels=tf.zeros_like(Dg)))
    d_loss = d_loss_real + d_loss_fake

# generator loss
with tf.name_scope('g_loss'):

    g_loss_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logits, labels=tf.ones_like(Dg)))

    g_loss_patho = StyleFeature.get_patho_loss(img_detector, syn_detector)

    g_loss_severity = StyleFeature.get_severity_loss(img_detector, syn_detector)

    g_loss_retinal = StyleFeature.get_retinal_loss(img, syn, mask)
   
    g_loss_tv = StyleFeature.get_tv_loss(syn, mask) #  + maskedtv_weight * StyleFeature.get_tv_loss(syn, mask, tv_mask)
    
    g_loss = w_adv * g_loss_adversarial \
             + w_patho * g_loss_patho \
             + w_severity * g_loss_severity \
             + w_retinal * g_loss_retinal \
             + w_tv * g_loss_tv

    
# split the variable for two differentiable function
t_vars = tf.trainable_variables()
d_vars = list(set(t_vars) & set(tf.global_variables('discriminator')))
g_vars = list(set(t_vars) & set(tf.global_variables('generator')))

# optimizer
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('train'):
    d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*0.4).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

# =============================== summary prepare ============================================= #

# write summary
Dx_sum = tf.summary.histogram("Dx", Dx)
Dg_sum = tf.summary.histogram("Dg", Dg)

Dx_sum_scalar = tf.summary.scalar("Dx_value", tf.reduce_mean(Dx))
Dg_sum_scalar = tf.summary.scalar("Dg_value", tf.reduce_mean(Dg))

# syn_sum = tf.image_summary("synthesize", syn)

d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
d_loss_sum = tf.summary.scalar("d_loss", d_loss)
g_loss_patho_sum = tf.summary.scalar("g_loss_patho", g_loss_patho)
g_loss_severity_sum = tf.summary.scalar("g_loss_severity", g_loss_severity)
g_loss_retinal_sum = tf.summary.scalar("g_loss_retinal", g_loss_retinal)
g_loss_tv_sum = tf.summary.scalar("g_loss_tv", g_loss_tv)
g_loss_adversarial_sum = tf.summary.scalar("g_loss_adversarial", g_loss_adversarial)
g_loss_sum = tf.summary.scalar("g_loss", g_loss)

# g_sum = tf.merge_summary([Dg_sum, syn_sum, d_loss_fake_sum, g_loss_sum])
# d_sum = tf.merge_summary([Dx_sum, d_loss_real_sum, d_loss_fake_sum, d_loss_sum])

# g_sum = tf.merge_summary([Dg_sum_scalar])
# d_sum = tf.merge_summary([Dx_sum_scalar])

sum_merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter(summary_dir,)  # graph=tf.get_default_graph())

# =============================== train phase ============================================= #
init = tf.variables_initializer(d_vars + g_vars)
sess = K.get_session()
saver = tf.train.Saver(d_vars + g_vars, max_to_keep=None)

sess.run(init)
#writer = tf.train.SummaryWriter(model_directory, sess.graph)

# ==================================== save initialization ================================ #
if load_model:
    ckpt = tf.train.get_checkpoint_state('Model_and_Result/' + load_model + '/models')
    saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.save(sess, model_directory + '/model-' + str(0) + '.cptk')
    print("load saved model and SAVE")
elif save_model:
    saver.save(sess, model_directory + '/model-' + str(0) + '.cptk')
    print("Saved begining Model ")

def augmented(db, q):
    for data_train, _ in db:
        for batch in data_train:
            xs, ys, ms = batch
            for i in range(batch_size):
                M = cv2.getRotationMatrix2D((np.random.randint(240, 272), np.random.randint(240, 272)),
                                            np.random.randint(-5, 5), np.random.uniform(0.97, 1.05))
                xs[i] = cv2.warpAffine(xs[i], M, (512, 512))
                ys[i] = cv2.warpAffine(ys[i], M, (512, 512))
                ms[i] = cv2.warpAffine(ms[i], M, (512, 512))
                # import IPython
                # IPython.embed()
            q.put((xs, ys, ms))
    q.put(None)


with open('DMB/{}.test_amaps_reconstruction'.format(real_img_test_dataset), 'rb') as file:
    test_amaps_reconstruction = pickle.load(file)

with open('data/'+real_img_test_dataset+'_test.list', 'r') as f:
    file_list = yaml.safe_load(f)
select = [file_list.index(args.test1), file_list.index(args.test2), file_list.index(args.test3), file_list.index(args.test4), ]

img_sample = np.load('data/{}_test_image.npy'.format(real_img_test_dataset))[select, ...]
gt_sample = np.load('data/{}_test_gt.npy'.format(real_img_test_dataset))[select, ..., [0]]
mask_sample = np.load('data/{}_test_mask.npy'.format(real_img_test_dataset))[select, ...]

img_sample = (np.reshape(img_sample, [-1, img_x, img_y, img_channel]) - 0.5) * 2.0
gt_sample = (np.reshape(gt_sample, [-1, img_x, img_y, gt_channel]) - 0.5) * 2.0
mask_sample = (np.reshape(mask_sample, [-1, img_x, img_y, 1]) - 0.5) * 2.0


# ==================================== start training ===================================== #
stime=time.time()
q = queue.Queue()
for epoch in range(max_epoch):
    batchNum = 1

    thread_aug = threading.Thread(target=augmented, args=[db, q])
    thread_aug.setDaemon(True)
    thread_aug.start()

    while True:
        batch = q.get()
        if batch is None:
            break
        else:
            img_id = batchNum - 1

            z_sample = np.random.normal(0, 0.001, size=[batch_size, z_size]).astype(np.float32)#mean
            zs = z_sample

            xs, ys, ms = batch
            if xs.shape[0] != batch_size:
                continue

            # xs = np.transpose(xs, (0, 2, 3, 1))
            xs = (np.reshape(xs, [batch_size, img_size, img_size, img_channel]) - 0.5) * 2.0
            xs = np.lib.pad(xs, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            # ys = np.transpose(ys, (0, 2, 3, 1))
            ys = ys[..., [0]]
            ys = (np.reshape(ys, [batch_size, img_size, img_size, gt_channel]) - 0.5) * 2.0
            ys = np.lib.pad(ys, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            ms = (np.reshape(ms, [batch_size, img_size, img_size, 1]) - 0.5) * 2.0
            ms = np.lib.pad(ms, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))

            feed_dict = {img: xs, gt: ys, z: zs, mask: ms}
            # Update the discriminator
            _, dLoss = sess.run([d_optimizer, d_loss], feed_dict=feed_dict)

            # Update the generator, twice for good measure.
            _ = sess.run([g_optimizer], feed_dict=feed_dict)
            

            # _, gLoss, advL, pathoL, retinalL, tvL = sess.run([g_optimizer, g_loss, g_loss_adversarial, g_loss_patho, g_loss_retinal, g_loss_tv], feed_dict=feed_dict)
            _, gLoss, advL, pathoL, severityL, retinalL, tvL, summ = sess.run([g_optimizer, g_loss, g_loss_adversarial, g_loss_patho, g_loss_severity, g_loss_retinal, g_loss_tv, sum_merged], feed_dict=feed_dict)

            sum_writer.add_summary(summ, training_util.global_step(sess, global_step))
            print("[Epoch: %2d.%2d / %2d] [%4d]G Loss: %.4f D Loss: %.4f, patho: %.4f, severity: %.4f, retinal: %.4f, adv: %.4f, tv: %.4f" \
                  % (epoch, img_id, max_epoch, batchNum, gLoss, dLoss, w_patho*pathoL, w_severity*severityL, w_retinal*retinalL, w_adv*advL, w_tv*tvL))
            with open(model_directory + '/training_log.txt', 'a') as text_file:
                text_file.write(
                    "[Epoch: %2d.%2d / %2d] [%4d]G Loss: %.4f D Loss: %.4f, patho: %.4f, severity: %.4f, retinal: %.4f, adv: %.4f, tv: %.4f \n"
                    % (epoch, img_id, max_epoch, batchNum, gLoss, dLoss, w_patho*pathoL, w_severity*severityL, w_retinal*retinalL, w_adv*advL, w_tv*tvL))
            batchNum += 1
            if training_util.global_step(sess, global_step) % 100 == 0:



                z1 = np.random.normal(0, 0.001, size=[1, z_size]).astype(np.float32)
                z2 = np.random.normal(0, 1.0, size=[1, z_size]).astype(np.float32)
                z3 = np.random.normal(0, 0.01, size=[1, z_size]).astype(np.float32)

                sa, sb, sc, sd = 0, 1, 2, 3


                syn_sample_a, dLreal_val_a, dLfake_val_a = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sa]], gt: [gt_sample[sa]], z: z1, mask:[mask_sample[sa]],
                                                                               act_input[64]: [test_amaps_reconstruction[sa][64]],
                                                                               act_input[256]: [test_amaps_reconstruction[sa][256]]
                                                                               })
                syn_sample_b, dLreal_val_b, dLfake_val_b = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sb]], gt: [gt_sample[sb]], z: z3, mask:[mask_sample[sb]],
                                                                               act_input[64]: [test_amaps_reconstruction[sb][64]],
                                                                               act_input[256]: [test_amaps_reconstruction[sb][256]]
                                                                               })
                syn_sample_c, dLreal_val_c, dLfake_val_c = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sc]], gt: [gt_sample[sc]], z: z2, mask:[mask_sample[sc]],
                                                                               act_input[64]: [test_amaps_reconstruction[sc][64]],
                                                                               act_input[256]: [test_amaps_reconstruction[sc][256]]
                                                                               })
                syn_sample_d, dLreal_val_d, dLfake_val_d = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sd]], gt: [gt_sample[sd]], z: zs[:1], mask:[mask_sample[sd]],
                                                                               act_input[64]: [test_amaps_reconstruction[sd][64]],
                                                                               act_input[256]: [test_amaps_reconstruction[sd][256]]
                                                                               })

                syn_sample = np.concatenate((syn_sample_a, syn_sample_b, syn_sample_c,syn_sample_d),axis=0)

                syn_sample_am = (syn_sample_a + 1) * ((mask_sample[sa] + 1) / 2) - 1
                syn_sample_bm = (syn_sample_b + 1) * ((mask_sample[sb] + 1) / 2) - 1
                syn_sample_cm = (syn_sample_c + 1) * ((mask_sample[sc] + 1) / 2) - 1
                syn_sample_dm = (syn_sample_d + 1) * ((mask_sample[sd] + 1) / 2) - 1
                syn_sample_m = np.concatenate((syn_sample_am, syn_sample_bm, syn_sample_cm, syn_sample_dm), axis=0)
                
                dLreal_val = (dLreal_val_a + dLreal_val_b + dLreal_val_c + dLreal_val_d) / 4
                dLfake_val = (dLfake_val_a + dLfake_val_b + dLfake_val_c + dLfake_val_d) / 4

                # Save sample generator images for viewing training progress.
                save_images(np.reshape(syn_sample, [sample_batch, img_x, img_y, img_channel]),
                            [int(np.sqrt(sample_batch)), int(np.sqrt(sample_batch))],
                            sample_directory + '/fig' + str(training_util.global_step(sess, global_step)) + '.png')
                            
                save_images(np.reshape(syn_sample_m, [sample_batch, img_x, img_y, img_channel]),
                            [int(np.sqrt(sample_batch)), int(np.sqrt(sample_batch))],
                            sample_directory2 + '/fig' + str(training_util.global_step(sess, global_step)) + '.png')

                print("[Sample (global_step = %d)] real: %.4f fake: %.4f" \
                      % (training_util.global_step(sess, global_step), np.mean(dLreal_val), np.mean(dLfake_val)))
                with open(model_directory + '/training_log.txt', 'a') as text_file:
                    text_file.write("[Sample (global_step = %d)] real: %.4f fake: %.4f \n"
                                    % (training_util.global_step(sess, global_step), np.mean(dLreal_val),
                                       np.mean(dLfake_val)))

            if training_util.global_step(sess, global_step) % 30000 == 0:
                saver.save(sess,
                           model_directory + '/model-' + str(training_util.global_step(sess, global_step)) + '.cptk')
                print("Saved Model %d, time: %.4f" % (training_util.global_step(sess, global_step), time.time()-stime))

             

saver.save(sess, model_directory + '/model-' + str(training_util.global_step(sess, global_step)) + '.cptk')
print("Saved Model %d, time: %.4f" % (training_util.global_step(sess, global_step), time.time()-stime))

sess.close()

