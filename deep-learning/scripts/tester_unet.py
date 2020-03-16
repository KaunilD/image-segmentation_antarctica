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

from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


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

        image_out = np.zeros((h_, w_, c))
        image_out[:h, :w, :] = image_in
        return image_out

    def get_tiles(self, image):
        i_tiles = []
        image = GTiffDataset.pad_image(image, self.tile_size)
        height, width, c = image.shape


        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                print("breaking")
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
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    return model

if __name__=="__main__":

    root_dir = 'C:/Users/dhruv/Development/git/independent-study/data/pre-processed/dryvalleys/WV03/'
    stride = 256
    tile_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = createUNet()
    if torch.cuda.device_count() >=1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)
    model.load_state_dict(torch.load("C:/Users/dhruv/Development/git/independent-study/deep-learning/models/unet-sgd-4-0.9-4-25.pth")["model"])
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
        mask+=0.5
        height, width, _ = image.shape
        for i in range(1, height, stride):
            if i+tile_size > height:
                break
            for j in range(1, width, stride):
                output = outputs[counter]
                output = np.moveaxis(output, 0, -1)
                mask[
                    i-1:i+tile_size-1,
                    j-1:j+tile_size-1,
                    :2
                ] = np.copy(output)
                counter+=1

        #mask = mask[:, :, 1] > mask[:, :, 0]

        plt.imsave(
            "{}_{}.png".format( os.path.basename(img).split(".")[0] , "unet" ),
            mask)
