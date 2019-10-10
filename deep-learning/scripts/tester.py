import torch
import cv2
from PIL import Image
import numpy as np
import glob
import math
import os
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import deeplab
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None

class GTiffDataset(torch_data.Dataset):
    def __init__(self,
                 root_dir, split, tile_size = 256, stride = 256, debug = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tile_size = tile_size
        self.stride = stride
        self.root_dir = root_dir
        self.transform = transform
        self.debug = debug
        self.images = self.read_dir()

    def get_tiles(self, image):
        i_tiles = []
        width = image.shape[2] - image.shape[2]%self.tile_size
        height = image.shape[1] - image.shape[1]%self.tile_size

        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                break
            for j in range(0, width, self.stride):
                img_tile = image[
                    :,
                    i:i+self.tile_size,
                    j:j+self.tile_size
                ]

                i_tiles.append(img_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)

        return i_tiles

    def read_dir(self):
        tiles = []
        images = sorted(glob.glob(self.root_dir + '/' + '*_3031.tif'))
        for idx, img in enumerate(images):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(images)))
            image = Image.open(img)
            image = np.asarray(image)
            image = np.moveaxis(image, 2, 0)
            print(image.shape)
            i_tiles = self.get_tiles(image)
            for im in i_tiles:
                tiles.append(im)
            print(len(tiles))
            del image

        print()
        return tiles

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.images[idx]
        return sample

def test(model, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    outputs = []
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image = sample.float()
            image = image.to(device)
            outputs.append(model(image))
            tbar.set_description('{}%'.format(int((i/num_samples)*100)))

    return outputs

if __name__=="__main__":
    root_dir = '../../data/pre-processed/dryvalleys/WV02'
    stride = 256
    tile_size = 256

    device = torch.device("cuda")

    model = deeplab.DeepLab(output_stride=16)
    model.load_state_dict(torch.load("../../models/deeplabv3-resnet-bn2d-1.pth")["model"])
    model.to(device)

    gtiffdataset = GTiffDataset(root_dir, split='test', stride=256, debug=False)
    test_dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=1)
    outputs = test(model, device, test_dataloader)

    images = sorted(glob.glob(root_dir + '/' + '*_3031.tif'))

    for idx, img in enumerate(images):
        counter = 0

        image = Image.open(img)
        image = np.asarray(image)
        shape = image.shape

        mask = np.zeros((shape[0], shape[1]))

        width = mask.shape[1] - mask.shape[1]%tile_size
        height = mask.shape[0] - mask.shape[0]%tile_size

        for i in range(0, height, stride):
            if i+tile_size > height:
                break
            for j in range(0, width, stride):

                img_tile = image[
                    i:i+tile_size,
                    j:j+tile_size,
                    :
                ]
                counter+=1

                if np.sum(np.sum(np.sum(img_tile))) == 0:
                    continue

                output = outputs[counter].cpu().numpy()
                mask_tile = output[0][1]>output[0][0]

                mask_tile = np.reshape(mask_tile, (256, 256))

                mask[
                    i:i+tile_size,
                    j:j+tile_size
                ] = np.copy(mask_tile)

        plt.imsave(str(idx)+".png", mask)
