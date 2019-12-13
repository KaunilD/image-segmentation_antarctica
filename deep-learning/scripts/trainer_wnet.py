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
from models import wnet

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

                if np.sum(np.sum(np.sum(img_tile))) == 0 or np.prod(mask_tile.shape) != self.tile_size*self.tile_size:
                    continue

                i_tiles.append(img_tile)
                m_tiles.append(mask_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_mask.png", mask_tile)

        return i_tiles, m_tiles
        # Calcium@20
    def read_dir(self):
        tiles = [[], []]
        for idx, [img, msk] in enumerate(zip(self.root_dir[0], self.root_dir[1])):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir[0])))

            image = Image.open(img)
            mask = Image.open(msk)
            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM))

            cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = np.moveaxis(image, 2 ,0)

            mask = np.asarray(mask)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            i_tiles, m_tiles = self.get_tiles(image, mask)
            for im, ma in zip(i_tiles, m_tiles):
                tiles[0].append(im)
                tiles[1].append(ma)
            print(len(tiles[0]))
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


def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc')
    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model

def train(model, optimizer, criterion, device, dataloader, psi=0.5):
    model.train()
    train_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)

        enc = model(image, returns='enc')
        n_cut_loss=gradient_regularization(enc)*psi
        n_cut_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        dec = model(image, returns='dec')
        rec_loss=torch.mean(torch.pow(torch.pow(image, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
        rec_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += rec_loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, criterion, device, dataloader, psi=0.5):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.float().to(device)

            enc, dec = model(image, returns="both")
            rec_loss=torch.mean(torch.pow(torch.pow(image, 2) + torch.pow(dec, 2), 0.5))*(1-psi)

            val_loss += rec_loss.item()
            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
    return val_loss

if __name__=="__main__":

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    model_save_pth = '../models'
    epochs = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    images_list = sorted(glob.glob('../../data/pre-processed/dryvalleys/WV02/' + '*_3031.tif'))
    masks_list = sorted(glob.glob('../../data/pre-processed/dryvalleys/WV02/' + '*_3031_mask.tif'))


    gtiffdataset = GTiffDataset(
        [images_list[:100], masks_list[:100]],
        tile_size=256, split='train', stride=256, debug=False)

    val_gtiffdataset = GTiffDataset(
        [images_list[100:], masks_list[100:]],
        tile_size=256, split='val', stride=256, debug=False)

    train_dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_gtiffdataset, num_workers=0, batch_size=64)


    model = wnet.WNet(4)

    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.Adam(lr=0.005, weight_decay=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters())
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, verbose=True
    )


    criterion = focal_loss
    train_log = []
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, train_dataloader)

        scheduler.step(val_loss)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        model_save_str = '{}/{}-{}-{}-{}.{}'.format(
            model_save_pth, model.module.name,
            "-", "bn2d", epoch, "pth"
        )

        torch.save(
            state,
            model_save_str
        )
        train_log.append([train_loss, val_loss])
        np.save("train_log_{}".format(model.module.name), train_log)
        print(epoch, train_loss, val_loss)
