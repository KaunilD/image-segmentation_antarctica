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
        self.images, self.masks = self.read_dir()

    def get_tiles(self, image, mask):
        i_tiles, m_tiles = [], []
        width = image.shape[1] - image.shape[1]%self.tile_size
        height = image.shape[0] - image.shape[0]%self.tile_size

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
                    :,
                    i:i+self.tile_size,
                    j:j+self.tile_size
                ]

                if np.prod(mask_tile.shape)!= 256*256 or np.prod(img_tile.shape)!= 3*256*256:
                    print("error")

                if np.sum(img_tile>0) != 3*self.tile_size*self.tile_size or np.prod(mask_tile.shape) != self.tile_size*self.tile_size:
                    continue

                i_tiles.append(img_tile)
                m_tiles.append(mask_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)
                    plt.imsave("debug/" + str(i) + "_" + str(j) + "_mask.png", mask_tile)

        return i_tiles, m_tiles
        # Calcium@20
    def read_dir(self):
        tiles = [[], []]
        for idx, [img, msk] in enumerate(zip(self.root_dir[0], self.root_dir[1])):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir[0])))

            image = Image.open(img)
            mask = Image.open(msk)

            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8)

            mask = np.asarray(mask, dtype=np.float32)
            mask = np.reshape(mask, (1,)+mask.shape)
            mask/=255.0

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
            return [self.transform(self.images[idx]), self.masks[idx]]
        return [self.images[idx], self.masks[idx]]

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
    f1_metric = 0.0

    tbar = tqdm(dataloader)

    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)

        output = model(image)

        y_pred = output['out'].data.cpu().numpy().ravel()
        y_true = target.data.cpu().numpy().ravel()
        f1_metric += f1_score(y_true > 0, y_pred > 0.1)

        loss = criterion(output['out'], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return f1_metric/(num_samples)

def validate(model, criterion, device, dataloader):
    model.eval()
    f1_metric = 0.0
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.float().to(device)

            output = model(image)

            y_pred = output['out'].data.cpu().numpy().ravel()
            y_true = target.data.cpu().numpy().ravel()
            f1_metric += f1_score(y_true > 0, y_pred > 0.1)

            loss = criterion(output['out'], target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (train_loss / (i + 1)))
    return f1_metric/(num_samples)

def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
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
    masks_lists = [
        "1040010006B14C00_3031_mask.tif","10400100466E0A00_3031_mask.tif","10400100355E0900_3031_mask.tif",
        "1040010038223D00_3031_mask.tif","1040010029761800_3031_mask.tif","1040010029AB0800_3031_mask.tif",
        "104001002722CB00_3031_mask.tif","104001002722CB00_3031_mask.tif","104001002722CB00_3031_mask.tif",
        "104001002642C800_3031_mask.tif","104001002642C800_3031_mask.tif","104001002642C800_3031_mask.tif",
        "104001002642C800_3031_mask.tif","104001000647F000_3031_mask.tif","104001000647F000_3031_mask.tif",
        "1040010006846000_3031_mask.tif","10400100467A6F00_3031_mask.tif","104001004664AD00_3031_mask.tif"

    ]

    images_list = [path + '/' + i for i in images_list]
    masks_list = [path + '/' + i for i in masks_list]

    return images_list, masks_list

def create_args():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation of satellite imagery from antractica \
        using Pretrained DeepLabV3 model with Resnet 101 backbone"
    )
    parser.add_argument(
        "--root-dir",
        default="/projects/kadh5719/image-segmentation_antarctica",
        type=str,
        help="root directory of the project.",
    )
    parser.add_argument(
        "--data-dir",
        default="/projects/kadh5719/image-segmentation_antarctica/data/pre-processed/dryvalleys/WV03",
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
        default=30,
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
        default="/projects/kadh5719/image-segmentation_antarctica/deep-learning/models/",
        type=str,
        help="path to save models in.",
    )

    return parse.parse_args()

if __name__=="__main__":

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    args = create_args()

    epochs = args.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images_list, masks_list = get_dataset(args.data_dir)
    train_len = int(args.train_test_split * len(images_list))

    gtiffdataset = GTiffDataset(
        [images_list[:train_len], masks_list[:train_len]],
        tile_size=256, split='train', stride=256,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        debug=False)

    val_gtiffdataset = GTiffDataset(
        [images_list[train_len:], masks_list[train_len:]],
        tile_size=256, split='val', stride=256,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        debug=False)

    train_dataloader = torch_data.DataLoader(
        gtiffdataset, num_workers=0, batch_size=args.train_batch_size, drop_last=True)
    val_dataloader = torch_data.DataLoader(
        val_gtiffdataset, num_workers=0, batch_size=args.test_batch_size, drop_last=True)

    model = createDeepLabv3()
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)


    optimizer = torch.optim.SGD(lr=args.lr, momentum=0.9, weight_decay = 1e-4,
        params= model.parameters()
    )

    if args.restart_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])

        model.to(device)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr

    criterion = torch.nn.MSELoss(reduction='mean')

    train_log = []
    for epoch in range(args.epoch):

        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, val_dataloader)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        model_save_str = '{}/{}-{}.{}'.format(
            args.checkpoint_save_path, args.checkpoint_prefix, epoch, "pth")

        torch.save(state, model_save_str)

        train_log.append([train_loss, val_loss])

        np.save("train_log_".format(args.checkpoint_prefix), train_log)

    print(epoch, train_loss, val_loss)
