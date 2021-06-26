import numpy as np
import cv2
import glob
import yaml
import os

dataset_name='IDRiD'
for mode in ['train', 'test']:
    filelist = sorted(glob.glob(mode+'_512/*.jpg'))

    images = np.stack(list(map(cv2.imread, filelist)))
    images = images/255.
    images = images[..., ::-1]  # BGR2RGB
    images = images.astype('float32')
    print('images', images.shape, images.dtype)
    np.save('../'+dataset_name+'_'+mode+'_image.npy', images)

    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    mask = mask/255.
    mask = mask.astype('float32')
    mask = np.tile(mask[None, ...], (images.shape[0], 1, 1))
    print('mask', mask.shape, mask.dtype)
    np.save('../'+dataset_name+'_'+mode+'_mask.npy', mask)

    gt1 = np.stack(list(map(lambda x: cv2.imread(x.replace('.jpg', '_VS.png'), cv2.IMREAD_GRAYSCALE), filelist)))
    gt1 = gt1/255.
    gt1 = gt1.astype('float32')
    # gt2 = np.load('../../Visualization/'+dataset_name+'_'+mode+'_dump.npy')
    # gt = np.concatenate([gt1[..., None], gt2], 3)
    # print('gt1', gt1.shape, gt1.dtype, 'gt2', gt2.shape, gt2.dtype, 'gt', gt.shape, gt.dtype)
    np.save('../'+dataset_name+'_'+mode+'_gt.npy', gt1[..., None])

    # dump file_name list
    with open('../'+dataset_name+'_'+mode+'.list', 'w') as f:
        yaml.dump(list(map(os.path.basename, filelist)), f)
