import torch
import cv2
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
        self.images, self.masks = self.read_dir()

    def get_tiles(self, image, mask):
        i_tiles, m_tiles = [], []
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

                mask_tile = mask[
                    i:i+self.tile_size,
                    j:j+self.tile_size
                ]

                i_tiles.append(img_tile)
                m_tiles.append(mask_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_mask.png", mask_tile)
        return i_tiles, m_tiles

    def read_dir(self):
        tiles = [[], []]
        images = sorted(glob.glob(self.root_dir + '/' + '101001000A4E4B00_4326_cropped.png'))
        masks = sorted(glob.glob(self.root_dir + '/' + '101001000A4E4B00_mask_4326.tif'))
        for idx, [i, m] in enumerate(zip(images, masks)):
            print('Reading item # {} - {}/{}'.format(i, idx+1, len(images)))
            image = cv2.imread(i)
            mask = cv2.imread(m, 0)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            image = np.moveaxis(image, 2, 0)

            i_tiles, m_tiles = self.get_tiles(image, mask)
            for im, ma in zip(i_tiles, m_tiles):
                tiles[0].append(im)
                tiles[1].append(ma)
            del image
            del mask
        print()
        return tiles

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.masks[idx]]
        return sample

def focal_loss(output, target, device, gamma=2, alpha=0.5):
    n, c, h, w = output.size()
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    logpt = -criterion(output, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    loss /= n

    return loss

def train(model, optimizer, criterion, device, dataloader):
    model.train()
    train_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target, device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    epochs = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gtiffdataset = GTiffDataset('../../data/pre-processed/dryvalleys/WV02', split='train', stride=128, debug=False)
    val_gtiffdataset = GTiffDataset('../../data/pre-processed/dryvalleys/QB02', split='val', stride=128, debug=False)
    train_dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=4)
    model = deeplab.DeepLab(output_stride=16)
    model.to(device)
    optimizer = torch.optim.SGD(
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        params=model.get_1x_lr_params()
    )

    criterion = focal_loss


    train(model, optimizer, criterion, device, train_dataloader)
