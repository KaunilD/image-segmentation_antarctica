import pickle
import cv2
from PIL import Image
import numpy as np
import glob
import math
import os
import sys
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
        width = image.shape[2] - image.shape[2]%self.tile_size
        height = image.shape[1] - image.shape[1]%self.tile_size

        for i in range(0, height, self.stride):
            if i+self.tile_size > height:
                break
            for j in range(0, width, self.stride):
                img_tile = image[
                    :,
                    i:i+self.tile_size,
                    j:j+self.tile_size,
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

            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)

            image/=255.0
            image = np.moveaxis(image, -1, 0)

            mask = np.asarray(mask, dtype=np.float32)
            mask = np.reshape(mask, (1,)+mask.shape)
            mask/=255.0

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
        if self.transform:
            print(self.images[idx].shape)
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
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output['out'], target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, criterion, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.float().to(device)

            output = model(image)
            loss = criterion(output['out'], target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (train_loss / (i + 1)))
    return val_loss

def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

if __name__=="__main__":

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    model_save_pth = '../models'
    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    images_list = sorted(glob.glob('../../data/pre-processed/dryvalleys/WV03/' + '*_3031.tif'))
    masks_list = sorted(glob.glob('../../data/pre-processed/dryvalleys/WV03/' + '*_3031_mask.tif'))
    with open("images.pickle", "wb") as file_:
        pickle.dump(images_list, file_)
    """
    images_list = [
        "../../data/pre-processed/dryvalleys/WV03/1040010006B14C00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100466E0A00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100355E0900_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010038223D00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010029761800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010029AB0800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001000647F000_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001000647F000_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010006846000_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100467A6F00_3031.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001004664AD00_3031.tif"
    ]
    masks_list = [
        "../../data/pre-processed/dryvalleys/WV03/1040010006B14C00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100466E0A00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100355E0900_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010038223D00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010029761800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010029AB0800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002722CB00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001002642C800_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001000647F000_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001000647F000_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/1040010006846000_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/10400100467A6F00_3031_mask.tif",
        "../../data/pre-processed/dryvalleys/WV03/104001004664AD00_3031_mask.tif"

    ]
    train_len = int(0.8*len(images_list))
    #sys.exit(0)
    gtiffdataset = GTiffDataset(
        [images_list[:1], masks_list[:1]],
        tile_size=256, split='train', stride=256,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        debug=False)
    #sys.exit(0)
    val_gtiffdataset = GTiffDataset(
        [images_list[1:2], masks_list[1:2]],
        tile_size=256, split='val', stride=256, debug=False)

    train_dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=64, drop_last=True)
    val_dataloader = torch_data.DataLoader(val_gtiffdataset, num_workers=0, batch_size=64, drop_last=True)


    model = createDeepLabv3()


    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)
    model.module.name = "deeplabv3_pretrained"
    model.to(device)

    optimizer = torch.optim.SGD(lr=1e-4,
        params= model.parameters()
    )

    criterion = torch.nn.MSELoss(reduction='mean')
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    train_log = []
    for epoch in range(epochs):

        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, train_dataloader)

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
