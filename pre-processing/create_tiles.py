import numpy as np
import glob
import math
import os
import sys
import cv2
from PIL import Image

tile_size = 256
stride = 128
root_dir = '../data/pre-processed/dryvalleys/WV02'
counter = 0

images = sorted(glob.glob(root_dir + '/' + '*_3031.tif'))
masks = sorted(glob.glob(root_dir + '/' + '*_3031_mask.tif'))

for idx, [img, msk] in enumerate(zip(images, masks)):
    print('Reading item # {} - {}/{}'.format(img, idx+1, len(images)))
    image = Image.open(img)
    mask = Image.open(msk)
    image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM))
    #image = np.moveaxis(image, 0, -1)

    mask = np.asarray(mask)


    width = image.shape[1] - image.shape[1]%tile_size
    height = image.shape[0] - image.shape[0]%tile_size

    for i in range(0, height, stride):
        if i+tile_size > height:
            break
        for j in range(0, width, stride):
            img_tile = image[
                i:i+tile_size,
                j:j+tile_size,
                :
            ]

            mask_tile = mask[
                i:i+tile_size,
                j:j+tile_size
            ]

            if np.sum(np.sum(mask_tile)) <= tile_size*tile_size:
                continue
                
            cv2.imwrite("{}/tiles/image/{}.png".format(root_dir, counter), img_tile)
            cv2.imwrite("{}/tiles/mask/{}.png".format(root_dir, counter), mask_tile)

            counter += 1
