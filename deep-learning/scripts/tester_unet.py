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

import matplotlib.pyplot as plt

from models_pytorch.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


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
    @staticmethod
    def pad_image(image_in, tile_size):

        h, w, c = image_in.shape

        w_ = (w // tile_size + 1)*tile_size
        h_ = (h // tile_size + 1)*tile_size

        image_out = np.zeros((w_, h_, c))
        image_out[:h, :w, :] = image_in
        return image_out

    def get_tiles(self, image):
        i_tiles = []
        image = pad_image(image, self.tile_size)
        height, width, c = image.shape


        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                break
            for j in range(0, width, self.stride):
                img_tile = image[
                    i:i+self.tile_size,
                    j:j+self.tile_size,
                    :
                ]

                i_tiles.append(img_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)

        return i_tiles

    def read_dir(self):
        tiles = []

        for idx, img in enumerate(self.root_dir):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir)))
            image = Image.open(img)
            image = np.asarray(image, dtype=np.uint8)
            m = np.mean(image, axis=(0, 1, 2))
            s = np.std(image, axis=(0, 1, 2))
            image = (image - m) / s

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
        if self.transform:
            return self.transform(self.images[idx])
        return sample

def test(model, device, dataloader):
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    outputs = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image = sample.float()
            image = image.to(device)
            out = model(image)
            for jdx, j in enumerate(out):
                outputs.append(j.cpu().numpy())
            tbar.set_description('{}%'.format(int((i/num_samples)*100)))

    return outputs


def createUNet(outputchannels=1):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False)
    return model

if __name__=="__main__":

    root_dir = '../../data/pre-processed/dryvalleys/WV03/'
    stride = 256
    tile_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = createUNet()
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)
    model.load_state_dict(torch.load("../models/unet-sgd-4-0.9-4-25.pth")["model"])
    model.to(device)
    model.eval()

    images = sorted(glob.glob(root_dir + '/' + '10400100467A6F00_3031.tif'))[0:1]
    print(images)

    gtiffdataset = GTiffDataset(
        images, split='test', stride=stride,
        tile_size=tile_size, debug=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_dataloader = torch_data.DataLoader(
        gtiffdataset,
        num_workers=0,
        batch_size=8

    )
    outputs = test(model, device, test_dataloader)


    for idx, img in enumerate(images):
        counter = 0

        image = Image.open(img)
        image = np.asarray(image, dtype=np.uint8)
        image = GTiffDataset.pad_image(image, tile_size)
        mask = np.zeros(image.shape, dtype=np.float32)

        width = mask.shape[1] - mask.shape[1]%tile_size
        height = mask.shape[0] - mask.shape[0]%tile_size

        for i in range(0, height, stride):
            if i+tile_size > height:
                break
            for j in range(0, width, stride):

                img_tile = image[
                    i:i+tile_size,
                    j:j+tile_size
                ]

                counter+=1

                if np.sum(img_tile>0) != 3*tile_size*tile_size:
                    continue

                output = outputs[counter]
                output = np.moveaxis(output, 0, -1)

                mask[
                    i:i+tile_size,
                    j:j+tile_size,
                    :2
                ] = np.copy(output)
        #mask = mask[:, :, 1] > mask[:, :, 0]

        plt.imsave(
            "{}_{}.png".format( os.path.basename(img).split(".")[0] , "unet" ),
            mask)
