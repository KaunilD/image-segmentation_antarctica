import pickle
import cv2
from PIL import Image
import numpy as np
import glob
import math
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score

import torch
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

Image.MAX_IMAGE_PIXELS = None

class GTiffDataset(torch_data.Dataset):
    def __init__(self, root_dir, split, tile_size = 256, stride = 256, debug = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tile_size = tile_size
        self.split = split
        self.stride = stride
        self.root_dir = root_dir
        self.transform = transform
        self.debug = debug
        self.images, self.masks = self.read_dir()

    def get_tiles(self, image, mask):
        i_tiles, m_tiles = [], []
        width = image.shape[1] - image.shape[1]%self.tile_size
        height = image.shape[0] - image.shape[0]%self.tile_size
        
        print(image.shape, mask.shape, width, height)
        
        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                break
            for j in range(0, width, self.stride):
                img_tile = image[
                    i:i+self.tile_size,
                    j:j+self.tile_size,
                    :
                ]

                mask_tile = mask[
                    i:i+self.tile_size,
                    j:j+self.tile_size,
                    :
                ]

                if np.prod(mask_tile.shape)!= 256*256 or np.prod(img_tile.shape)!= 3*256*256:
                    print("error")
                """
                if np.sum(img_tile>0) != 3*self.tile_size*self.tile_size or \
                np.prod(mask_tile.shape) != self.tile_size*self.tile_size:
                    continue
                """

                i_tiles.append(img_tile)
                m_tiles.append(mask_tile)

                if self.debug and self.split == "val":
                    # Debugging the tiles
                    
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_mask.png", mask_tile[:, :, 0])
        
        return i_tiles, m_tiles
    
    def read_dir(self):
        tiles = [[], []]
        for idx, [img, msk] in enumerate(zip(self.root_dir[0], self.root_dir[1])):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir[0])))

            image = Image.open(img)
            mask = Image.open(msk)

            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8)
            
            mask = np.asarray(mask, dtype=np.uint8)
            mask = np.reshape(mask, mask.shape+(1,))
            
            i_tiles, m_tiles = self.get_tiles(image, mask)
            for im, ma in zip(i_tiles, m_tiles):
                tiles[0].append(im)
                tiles[1].append(ma)
            print("{} tiles obtained.".format(len(tiles[0])))
            del image
            del mask

        print()
        return tiles

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image   = self.images[idx]
        mask    = self.masks[idx]
        if self.transform:
            return [
                self.transform["input"](image), 
                self.transform["target"](mask)
            ]
        return [self.images[idx], self.masks[idx]]

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss


def train(model, optimizer, device, dataloader):
    model.train()

    train_loss = 0.0

    tbar = tqdm(dataloader)

    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)

        output = model(image)

        loss = calc_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, device, dataloader):
    model.eval()
    
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.float().to(device)

            output = model(image)

            loss = calc_loss(output, target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
    return val_loss

def createUNet(outputchannels=1):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    return model

def get_dataset(parent_dir):
    images_list = [
        "1040010006B14C00_3031.tif","10400100466E0A00_3031.tif","10400100355E0900_3031.tif",
        "1040010038223D00_3031.tif","1040010029761800_3031.tif","1040010029AB0800_3031.tif",
        "104001002722CB00_3031.tif","104001002722CB00_3031.tif","104001002722CB00_3031.tif",
        "104001002642C800_3031.tif","104001002642C800_3031.tif","104001002642C800_3031.tif",
        "104001002642C800_3031.tif","104001000647F000_3031.tif","104001000647F000_3031.tif",
        "1040010006846000_3031.tif","10400100467A6F00_3031.tif","104001004664AD00_3031.tif"
    ]
    masks_list = [
        "1040010006B14C00_3031_mask.tif","10400100466E0A00_3031_mask.tif","10400100355E0900_3031_mask.tif",
        "1040010038223D00_3031_mask.tif","1040010029761800_3031_mask.tif","1040010029AB0800_3031_mask.tif",
        "104001002722CB00_3031_mask.tif","104001002722CB00_3031_mask.tif","104001002722CB00_3031_mask.tif",
        "104001002642C800_3031_mask.tif","104001002642C800_3031_mask.tif","104001002642C800_3031_mask.tif",
        "104001002642C800_3031_mask.tif","104001000647F000_3031_mask.tif","104001000647F000_3031_mask.tif",
        "1040010006846000_3031_mask.tif","10400100467A6F00_3031_mask.tif","104001004664AD00_3031_mask.tif"
    ]

    images_list = [parent_dir + '/' + i for i in images_list]
    masks_list = [parent_dir + '/' + i for i in masks_list]

    return images_list, masks_list

def create_args():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation of satellite imagery from antractica \
        using Pretrained UNet model"
    )
    parser.add_argument(
        "--root-dir",
        default="/home/kadh5719/development/git/independent-study",
        type=str,
        help="root directory of the project.",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/kadh5719/development/git/independent-study/data/pre-processed/dryvalleys/WV03",
        type=str,
        help="directory containing image and masks in *.tif and *_mask.tif scheme.",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="learning rate.",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="training epochs.",
    )

    parser.add_argument(
        "--train-batch-size",
        default=32,
        type=int,
        help="training batch size.",
    )
    parser.add_argument(
        "--test-batch-size",
        default=64,
        type=int,
        help="test batch size.",
    )
    parser.add_argument(
        "--train-test-split",
        default=0.8,
        type=float,
        help="train and test data split.",
    )
    parser.add_argument(
        "--restart-checkpoint",
        default=None,
        type=str,
        help="restart training from a checkpoint."
    )

    parser.add_argument(
        "--checkpoint-prefix",
        default="unet",
        type=str,
        help="checkpoint prefix."
    )
    parser.add_argument(
        "--checkpoint-save-path",
        default="/home/kadh5719/development/git/independent-study/deep-learning/models/bedrock",
        type=str,
        help="path to save models in.",
    )

    return parser.parse_args()

if __name__=="__main__":
    """
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()
    """
    args = create_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images_list, masks_list = get_dataset(args.data_dir)
    train_len = int(args.train_test_split * len(images_list))

    val_gtiffdataset = GTiffDataset(
        [images_list[train_len:], masks_list[train_len:]],
        tile_size=256, split='val', stride=256,
        transform={
            "input":transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.])
                ]),
            "target":transforms.Compose([
                    transforms.ToTensor()
                ])
        },
        debug=False)
    
    gtiffdataset = GTiffDataset(
        [images_list[:train_len], masks_list[:train_len]],
        tile_size=256, split='train', stride=256,
        transform={
            "input":transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.])
                ]),
            "target":transforms.Compose([
                    transforms.ToTensor()
                ])
        },
        debug=False)
    
    train_dataloader = torch_data.DataLoader(
        gtiffdataset, num_workers=0, batch_size=args.train_batch_size, drop_last=True)
    val_dataloader = torch_data.DataLoader(
        val_gtiffdataset, num_workers=0, batch_size=args.test_batch_size, drop_last=True)

    model = createUNet()
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.Adam(lr=args.lr, params= model.parameters())

    if args.restart_checkpoint:
        print("Restrating from: {}".format(args.restart_checkpoint))
        checkpoint = torch.load(args.restart_checkpoint)
        model.load_state_dict(checkpoint["model"])

        model.to(device)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr

    criterion = torch.nn.MSELoss(reduction='mean')

    train_log = []
    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, device, train_dataloader)
        val_loss = validate(model, device, val_dataloader)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        model_save_str = '{}/{}-{}.{}'.format(
            args.checkpoint_save_path, args.checkpoint_prefix, epoch, "pth")

        torch.save(state, model_save_str)

        print(epoch, train_loss, val_loss)
        train_log.append([train_loss, val_loss])

        np.save("train_log_{}".format(args.checkpoint_prefix), train_log)

    print(epoch, train_loss, val_loss)
