"""Resize and crop images to square, save as tiff."""

import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image, ImageFilter
import cv2

N_PROC = 20

def convert(fname, crop_size, convert_fname):
    img = cv2.imread(fname)
    img_CW = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/cotton_wool_spots.png'))
    img_FP = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/fibrous_proliferation.png'))
    img_EX = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/hard_exudate.png'))
    img_MA = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/microaneurysm.png'))
    img_NS = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/neovascularization.png'))
    img_PH = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/preretinal_hemorrhage.png'))
    img_RH = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/retinal_hemorrhage.png'))
    img_VH = cv2.imread(fname.replace('images_896x896', 'lesion_segs_896x896').replace('.jpg', '/vitreous_hemorrhage.png'))

    img_CW = img_CW[..., 2] if img_CW is not None else img[..., 2] * 0
    img_FP = img_FP[..., 2] if img_FP is not None else img[..., 2] * 0
    img_EX = img_EX[..., 2] if img_EX is not None else img[..., 2] * 0
    img_MA = img_MA[..., 2] if img_MA is not None else img[..., 2] * 0
    img_NS = img_NS[..., 2] if img_NS is not None else img[..., 2] * 0
    img_PH = img_PH[..., 2] if img_PH is not None else img[..., 2] * 0
    img_RH = img_RH[..., 2] if img_RH is not None else img[..., 2] * 0
    img_VH = img_VH[..., 2] if img_VH is not None else img[..., 2] * 0

    ba = img
    h, w, _ = ba.shape

    if w > 1.2 * h:
        # to get the threshold, compute the maximum value of left and right 1/32-width part
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_max = gray[:, : w // 32].max().astype(int)
        right_max = gray[:, - w // 32:].max().astype(int)
        max_bg = np.maximum(left_max, right_max)

        # print(max_bg)  # TODO: DEBUG
        _, foreground = cv2.threshold(gray, max_bg + 20, 255, cv2.THRESH_BINARY)
        bbox = cv2.boundingRect(cv2.findNonZero(foreground))  # (x, y, width, height)

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, width, height = bbox

            # if we selected less than 80% of the original 
            # height, just crop the square
            if width < 0.8 * h or height < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(w, h)

    # do croping
    left, upper, width, height = bbox
    img = img[upper:upper+height, left:left+width, ...]
    img_CW = img_CW[upper:upper+height, left:left+width]
    img_FP = img_FP[upper:upper+height, left:left+width]
    img_EX = img_EX[upper:upper+height, left:left+width]
    img_MA = img_MA[upper:upper+height, left:left+width]
    img_NS = img_NS[upper:upper+height, left:left+width]
    img_PH = img_PH[upper:upper+height, left:left+width]
    img_RH = img_RH[upper:upper+height, left:left+width]
    img_VH = img_VH[upper:upper+height, left:left+width]

    #padding
    if width != height:
        if width > height:
            pad_width = width - height
            pad = ((pad_width//2, pad_width-pad_width//2), (0, 0))
        else:
            pad_width = height - width
            pad = ((0, 0), (pad_width // 2, pad_width - pad_width // 2))
        img = np.pad(img, (pad[0], pad[1], (0,0)), 'constant', constant_values=0)
        img_CW = np.pad(img_CW, pad, 'constant', constant_values=0)
        img_FP = np.pad(img_FP, pad, 'constant', constant_values=0)
        img_EX = np.pad(img_EX, pad, 'constant', constant_values=0)
        img_MA = np.pad(img_MA, pad, 'constant', constant_values=0)
        img_NS = np.pad(img_NS, pad, 'constant', constant_values=0)
        img_PH = np.pad(img_PH, pad, 'constant', constant_values=0)
        img_RH = np.pad(img_RH, pad, 'constant', constant_values=0)
        img_VH = np.pad(img_VH, pad, 'constant', constant_values=0)

    # resizing
    img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    img_CW = cv2.resize(img_CW, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_FP = cv2.resize(img_FP, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_EX = cv2.resize(img_EX, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_MA = cv2.resize(img_MA, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_NS = cv2.resize(img_NS, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_PH = cv2.resize(img_PH, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_RH = cv2.resize(img_RH, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    img_VH = cv2.resize(img_VH, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)


    cv2.imwrite(convert_fname, img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    cv2.imwrite(convert_fname.replace('.jpg', '_MASK.png'), binary)

    cv2.imwrite(convert_fname.replace('.jpg', '_CW.png'), img_CW)
    cv2.imwrite(convert_fname.replace('.jpg', '_FP.png'), img_FP)
    cv2.imwrite(convert_fname.replace('.jpg', '_EX.png'), img_EX)
    cv2.imwrite(convert_fname.replace('.jpg', '_MA.png'), img_MA)
    cv2.imwrite(convert_fname.replace('.jpg', '_NS.png'), img_NS)
    cv2.imwrite(convert_fname.replace('.jpg', '_PH.png'), img_PH)
    cv2.imwrite(convert_fname.replace('.jpg', '_RH.png'), img_RH)
    cv2.imwrite(convert_fname.replace('.jpg', '_VH.png'), img_VH)


def square_bbox(w, h):
    left = (w - h) // 2
    upper = 0
    right = left + h
    lower = h
    return (left, upper, right-left, lower-upper)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpg', extension).replace(directory,
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size, convert_fname)


def save(img, fname):
    img.save(fname, quality=97)

@click.command()
@click.option('--directory', default='retinal-lesions/images_896x896', show_default=True,
              help="Directory with original images.")
@click.option('--convert_directory', default='resized_512/', show_default=True,
              help="Where to save converted images.")
@click.option('--crop_size', default=512, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='jpg', show_default=True,
              help="Filetype of converted images.")
def main(directory, convert_directory, crop_size, extension):

    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg') or f.endswith('jpg') or f.endswith('png') or f.endswith('tiff')]
    filenames = sorted(filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 20
    batches = n // batchsize + 1
    pool = Pool(N_PROC)

    args = []

    for f in filenames:
        args.append((convert, (directory, convert_directory, f, crop_size, 
                           extension)))
        # break  # TODO: Debug

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()
