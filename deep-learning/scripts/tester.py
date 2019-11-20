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
from models import deeplab, uresnet
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

        for idx, img in enumerate(self.root_dir):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir)))
            image = Image.open(img)
            image = np.asarray(image)

            cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = np.moveaxis(image, 2 ,0)

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
            out = softmax(out)
            for jdx, j in enumerate(out):
                outputs.append(j.cpu().numpy())
            tbar.set_description('{}%'.format(int((i/num_samples)*100)))

    return outputs

if __name__=="__main__":
    """
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()
    """
    root_dir = '../../data/pre-processed/dryvalleys/WV02'
    stride = 256
    tile_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = deeplab.DeepLab()
    model = uresnet.UResNet()
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load("../../models/uresnet---bn2d-29.pth")["model"])
    model.to(device)
    model.eval()

    images = sorted(glob.glob(root_dir + '/' + '*_3031.tif'))[0:1]
    print(images)

    gtiffdataset = GTiffDataset(images, split='test', stride=256, debug=False)
    test_dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=8)
    outputs = test(model, device, test_dataloader)


    for idx, img in enumerate(images):
        counter = 0

        image = Image.open(img)
        image = np.asarray(image)
        shape = image.shape

        mask = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

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

                output = outputs[counter]
                output = np.moveaxis(output, 0, -1)

                mask[
                    i:i+tile_size,
                    j:j+tile_size,
                    :2
                ] = np.copy(output)
        mask = mask[:, :, 1] > mask[:, :, 0]

        plt.imsave(
            "{}_{}.png".format( os.path.basename(img).split(".")[0] , model.module.name ),
            mask)
