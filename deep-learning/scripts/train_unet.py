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

from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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
        self.split = split
        self.root_dir = root_dir
        self.transform = transform
        self.debug = debug
        self.images, self.masks = self.read_dir()

    def get_tiles(self, image, mask):
        i_tiles, m_tiles = [], []
        width = image.shape[1] - image.shape[1]%self.tile_size
        height = image.shape[0] - image.shape[0]%self.tile_size

        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                break
            for j in range(0, width, self.stride):
                if j+self.tile_size > width:
                    break
                img_tile = image[
                    i:i+self.tile_size,
                    j:j+self.tile_size,
                    :
                ]

                mask_tile = mask[
                    i:i+self.tile_size,
                    j:j+self.tile_size
                ]
                # atleast 30% pixels must be on
                if np.sum(mask_tile) <= 0.3*self.tile_size*self.tile_size:
                    continue

                i_tiles.append(img_tile)
                m_tiles.append(mask_tile)

                if self.debug:
                    # Debugging the tiles
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_mask.png", mask_tile)
        return i_tiles, m_tiles
    def read_dir(self):
        tiles = [[], []]
        for idx, [img, msk] in enumerate(zip(self.root_dir[0], self.root_dir[1])):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir[0])))

            image = Image.open(img)
            mask = Image.open(msk)

            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8)
            mask = np.asarray(mask, dtype=np.uint8)

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
        if self.transform:
            return [self.transform["input"](self.images[idx]), self.transform["target"](self.masks[idx])]
        return [self.images[idx], self.masks[idx]]


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

def ce_focal_loss(inputs, targets, alpha=1, gamma=0.5):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return torch.mean(F_loss)


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze()  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.sum()  # Or thresholded.mean() if you are interested in average across the batch



def train(model, optimizer, criterion, device, dataloader):
    model.train()

    train_loss = 0.0
    iou = 0.0

    tbar = tqdm(dataloader)

    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.to(device)

        output = model(image)

        iou += iou_pytorch(output > 0.5, target > 0.5)

        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return (iou/(num_samples)).item()

def validate(model, criterion, device, dataloader):
    model.eval()

    iou = 0.0
    val_loss = 0.0

    tbar = tqdm(dataloader)
    num_samples = len(dataloader)

    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.to(device)

            output = model(image)

            iou += iou_pytorch(output > 0.5, target > 0.5)

            loss = criterion(output, target)
            val_loss += loss.item()

            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
    return (iou/(num_samples)).item()

def createUNet(outputchannels=1):
    model = UNet(in_channels=3, out_channels=1, init_features=32)
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
        default=False,
        type=bool,
        help="restart training from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        default="model",
        type=str,
        help="checkpoint path."
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        help="checkpoint prefix."
    )
    parser.add_argument(
        "--checkpoint-save-path",
        default="/home/kadh5719/development/git/independent-study/deep-learning/models/",
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

    gtiffdataset = GTiffDataset(
        [images_list[:train_len], masks_list[:train_len]],
        tile_size=512, split='train', stride=512,
        transform={
        "input": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
        "target": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor()
                ])
        },
        debug=True)

    val_gtiffdataset = GTiffDataset(
        [images_list[train_len:], masks_list[train_len:]],
        tile_size=512, split='val', stride=512,
        transform={
        "input": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
        "target": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(30, shear=30),
                    transforms.ToTensor()
                ])
        },
        debug=True)

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
        print("Restrating from: {}".format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])

        model.to(device)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr

    train_log = []
    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, ce_focal_loss, device, train_dataloader)
        val_loss = validate(model, ce_focal_loss, device, val_dataloader)

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
