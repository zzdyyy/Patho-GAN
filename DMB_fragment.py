import cv2
import glob
import numpy as np
from scipy import ndimage
from random import randint
from StyleFeature import STYLE_LAYERS, STYLE_LAYERS_SIZE, STYLE_LAYERS_CHANNELS
import numpy as np

palette = [
    [182,182,254], [255,219,152], [168,255,153],

    [ 31, 119, 180],
    [255, 127,  14],
    [ 44, 160,  44],
    [214,  39,  40],
    [148, 103, 189],
    [140,  86,  75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189,  34],
    [ 23, 190, 207],

    [161, 201, 244],
    [255, 180, 130],
    [141, 229, 161],
    [255, 159, 155],
    [208, 187, 255],
    [222, 187, 155],
    [250, 176, 228],
    [207, 207, 207],
    [255, 254, 163],
    [185, 242, 240]
]
palette = np.array(palette, dtype='uint8')
np.random.seed(299792458)
palette = np.concatenate([palette, np.random.randint(256, size=[1000,3], dtype='uint8')])

def extract_descriptors(intermed_amap: np.ndarray, featmap: np.ndarray, seg_label: np.ndarray, 
                        dataset_name: str, img_id: int, fname_debug='debug/test.png'):
    """segment the lesions in featmap,
    predict the segment label by seg_label,
    return the descriptors (fragments) with its attributes"""

    height, width = featmap.shape[:2]

    # to gray scale
    gray = np.abs(featmap)
    gray = np.mean(gray, axis=2)
    gray = cv2.GaussianBlur(gray, (51, 51), 10)
    gray = gray / gray.max()
    cv2.imwrite(fname_debug.replace('.png', '_aSeg2.png'), gray * 255)

    # to binary and segmentation
    ret, binary = cv2.threshold((gray*255).astype('uint8'), 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.drawContours(binary*0, contours, -1, (255, 255, 255))
    cv2.imwrite(fname_debug.replace('.png', '_aSeg3.png'), img_contours)

    seg_label = (seg_label*255).astype('uint8')

    # extract each fragment and its attributs
    fragments = []
    for i in range(len(contours)):
        mask_i = cv2.drawContours(np.zeros([height, width], dtype=np.uint8),
                                           contours, i, (255, 255, 255), cv2.FILLED)
        cv2.imwrite(fname_debug.replace('.png', '_aSeg4.png'), mask_i)

        # compute scale and location
        original_scale = (mask_i > 0).sum() ** 0.5  # sqrt of the area
        (x_min, y_min), = contours[i].min(axis=0)
        (x_max, y_max), = contours[i].max(axis=0)

        # prepare intermed_amap fragment
        fragment_data = {}
        for amap_size, amap_data in intermed_amap.items():
            ratio = 512 // amap_size  # down sampling ratio
            mask_frag = cv2.resize(mask_i, (amap_size, amap_size), interpolation=cv2.INTER_LINEAR)

            fragment = amap_data[0].copy()  # fragment.shape=[256,256,32] [64, 64, 64]
            fragment[mask_frag==0] = 0.
            fragment = fragment[y_min//ratio:y_max//ratio+1, x_min//ratio:x_max//ratio+1]
            # cv2.imwrite(fname_debug.replace('.png', '_aSeg6.png'), (fragment[..., :3]/0.05+0.5)*255)
            fragment_data[amap_size] = fragment

        # debug: visualize seg_label
        # vis = np.zeros([height, width, 3], dtype='uint8')
        # vis[..., 2] = vis[..., 0] | seg_label[..., 0]
        # vis[..., 1] = vis[..., 1] | seg_label[..., 1]
        # vis[..., 1:] = vis[..., 1:] | seg_label[..., 2:3]
        # vis[..., :2] = vis[..., :2] | seg_label[..., 3:]
        # cv2.imwrite(fname_debug.replace('.png', '_aSeg5.png'), vis)

        # predict label of the predicted
        overlap = (seg_label & mask_i[..., None])
        overlap_score = (overlap>0).sum(axis=(0,1))
        predict_label = np.argmax(overlap_score)
        if overlap_score[predict_label] == 0:
            predict_label = -1


        fragments.append((x_min, y_min, x_max, y_max, original_scale,  # location, scale
                         fragment_data,  # data
                         dataset_name,
                         img_id,
                         predict_label,  # label
                         ))

    # debug: visulize with label
    # labels = ['MA','HE','EX','SE','UNK',]
    # colors = np.random.randint(0, 256, [len(contours), 3])
    # colors[17] = [182,182,254]; colors[11] = [255,219,152]; colors[1] = [168,255,153]
    # img_contours = cv2.imread('/home/nyh/tllt/IDRiD/test_512/IDRiD_69.jpg')  # np.zeros([512, 512, 3], 'uint8')
    # for i in range(len(contours)):
    #     img_contours = cv2.drawContours(img_contours, contours, i, tuple(colors[i].tolist()), 2)
    #     cv2.putText(img_contours,
    #                 labels[fragments[i][-1]],
    #                 (fragments[i][0], fragments[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, tuple(colors[i].tolist()), 1)
    # cv2.imwrite(fname_debug.replace('.png', '_aSeg3.add.png'), img_contours)

    return fragments


def rebuild_AMaps_by_cat(fragments, fragments_DB_by_cat, height=512, width=512, fname_debug='debug/test.png'):
    amap = {size: np.zeros([size, size, n_channel], dtype='float32')
            for size, n_channel in zip(STYLE_LAYERS_SIZE, STYLE_LAYERS_CHANNELS)}
    for x, y, category, seed, scale, rotation in fragments:
        seed = seed % len(fragments_DB_by_cat[category])
        x_min, y_min, x_max, y_max, original_scale,\
        feat_fragment, _, _, predict_label = fragments_DB_by_cat[category][seed]

        for size in STYLE_LAYERS_SIZE:
            ratio = height // size  # down sampling ratio

            # padding
            padding = (y//ratio, max(0, height//ratio - (y//ratio + y_max//ratio - y_min//ratio + 1))), \
                      (x//ratio, max(0, height//ratio - (x//ratio + x_max//ratio - x_min//ratio + 1))), \
                      (0, 0)
            feat_fragment_to_add = np.pad(feat_fragment[size], padding, 'constant')
            feat_fragment_to_add = feat_fragment_to_add[:height//ratio, :width//ratio, :]
            # cv2.imwrite(fname_debug.replace('.png', '_aSeg7.png'), (feat_fragment[size]/2+0.5) * 255)

            # rotating
            M_rot = cv2.getRotationMatrix2D(center=(x//ratio, y//ratio), angle=rotation, scale=scale)
            feat_fragment_to_add = cv2.warpAffine(feat_fragment_to_add, M_rot, (height//ratio, width//ratio))
            # cv2.imwrite(fname_debug.replace('.png', '_aSeg8.png'), (feat_fragment[size] / 2 + 0.5) * 255)

            amap[size] += feat_fragment_to_add

    return amap


def rebuild_AMaps_by_img(imgid, fragments_DB_by_img, height=512, width=512, fname_debug='debug/test.png',
                         randomize=False, quantity=None, multiple=None, lesion_map=None):
    """Reconstruct AMaps from descriptors that extracted from one reference image"""
    amap = {size: np.zeros([size, size, n_channel], dtype='float32')
            for size, n_channel in zip(STYLE_LAYERS_SIZE, STYLE_LAYERS_CHANNELS)}
    fragments = fragments_DB_by_img[imgid]
    if lesion_map is not None:  # set palette
        my_palette = palette[:len(fragments)]
        lesion_map = np.zeros([256, 256, 3], dtype='uint8')
    if quantity is not None:  # manipulate lesion quantity (0.x times)
        import random
        index = random.sample(range(len(fragments)), int(quantity*len(fragments)))
        if lesion_map is not None:
            my_palette = my_palette[index]
        fragments = [fragments[i] for i in index]
    if multiple is not None:  # manipulate lesion quantity (n times)
        fragments = fragments*multiple
    for fid, fragment in enumerate(fragments):
        x_min, y_min, x_max, y_max, original_scale, \
        feat_fragment, _, _, predict_label = fragment

        if randomize:
            new_x, new_y = 8*np.random.randint(0, 64), 8*np.random.randint(0, 64)
            x_min, x_max = new_x, new_x + x_max - x_min
            y_min, y_max = new_y, new_y + y_max - y_min

        for size in STYLE_LAYERS_SIZE:
            ratio = height // size

            # padding
            padding = (y_min//ratio, max(0, height//ratio - (y_max//ratio + 1))), \
                      (x_min//ratio, max(0, height//ratio - (x_max//ratio + 1))), \
                      (0, 0)
            feat_fragment_to_add = np.pad(feat_fragment[size], padding, 'constant')
            feat_fragment_to_add = feat_fragment_to_add[:height//ratio, :width//ratio, :]
            # cv2.imwrite(fname_debug.replace('.png', '_aSeg7.png'), (feat_fragment[size]/2+0.5) * 255)

            amap[size] += feat_fragment_to_add

            if lesion_map is not None and size == 256:
                lesion_map = lesion_map | (
                    my_palette[fid % len(my_palette)] &
                    (255*np.any(feat_fragment_to_add != 0, axis=2, keepdims=True)).astype('uint8')
                )

        ## DEBUG
        # def reg(x):
        #     return (x-x.min())/(x.max()-x.min())
        # cv2.imwrite(fname_debug.replace('.png', '_aSeg8.256.{}.png'.format(imgid)), reg(np.sum(amap[256], -1)) * 255)
        # cv2.imwrite(fname_debug.replace('.png', '_aSeg8.64.{}.png'.format(imgid)), reg(np.sum(amap[64], -1)) * 255)
    if lesion_map is not None:
        return amap, lesion_map
    return amap
