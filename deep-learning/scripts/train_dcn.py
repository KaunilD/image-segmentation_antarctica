import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler


import clustering
import models

import glob
import cv2
from models import uresnet
from models.vgg16 import vgg16
from models.segnet import SegNet
# Calcium@20
args = {
    "verbose": True,
    "batch": 16,
    "lr": 0.005,
    "wd":0.005
}


Image.MAX_IMAGE_PIXELS = None

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)





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

                if np.sum(np.sum(np.sum(img_tile))) == 0 or np.prod(img_tile.shape[1:]) != self.tile_size*self.tile_size:
                    continue

                i_tiles.append(img_tile)

                if self.debug:
                    # Debugging the tiles
                    img_tile = np.moveaxis(img_tile, 0, -1)
                    cv2.imwrite("debug/" + str(i) + "_" + str(j) + "_img.png", img_tile)

        return i_tiles
        # Calcium@20
    def read_dir(self):
        tiles = []
        for idx, img in enumerate(self.root_dir):
            print('Reading item # {} - {}/{}'.format(img, idx+1, len(self.root_dir)))

            image = Image.open(img)
            image = np.asarray(image.transpose(Image.FLIP_TOP_BOTTOM))

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


def compute_features(device, dataloader, model, N):
    if args["verbose"]:
        print('Compute features')
    end = time.time()
    model.eval()
    for i, input_tensor in enumerate(dataloader):
        input_var = input_tensor.float()
        input_var = input_var.to(device)
        flattened_input = input_var.view(input_var.size(0), -1)
        flattened_input = flattened_input.data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, flattened_input.shape[1]), dtype='float32')

        flattened_input = flattened_input.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args["batch"]: (i + 1) * args["batch"]] = flattened_input
        else:
            # special treatment for final batch
            features[i * args["batch"]:] = flattened_input


        # measure elapsed time
        end = time.time()

        if args["verbose"] and (i % 200) == 0:
            print('{} / {}\t'
                  'Time: {}'
                  .format(i, len(dataloader), end))
    return features


def train(device, loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    # switch to train mode
    model.train()
    losses = []

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        print(input_tensor.size(), target.size())

        n = len(loader) * epoch + i

        input_var = input_tensor.float()
        input_var = input_var.to(device)

        target_var = target.long()
        target_var = target_var.to(device)

        output = model(input_var)
        loss = crit(output, target_var)

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        losses.append(loss.item())
        """print('Epoch: [{}][{}/{}]\t'
              'Loss: {}'
              .format(epoch, i, len(loader), losses[-1]))
        """
    return np.mean(losses)



def main():

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SegNet(input_channels=3, output_channels=1)
    model.to(device)

    optimizer = torch.optim.Adam(lr=0.005, weight_decay=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters())
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    images_list = sorted(glob.glob('../../data/pre-processed/dryvalleys/WV02/' + '*_3031.tif'))
    # 115469.125 75.99947928705643
    gtiffdataset = GTiffDataset(
        images_list[:1],
        tile_size=256, split='train', stride=256, debug=False)

    dataloader = torch_data.DataLoader(gtiffdataset, num_workers=0, batch_size=16)

    deepcluster = clustering.Kmeans(2)

    # get the features for the whole dataset
    # basically, flattening the images.
    features = compute_features(device, dataloader, model, len(gtiffdataset))


    if args["verbose"]:
        print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=args["verbose"])
    


    for epoch in range(0, 30):
        end = time.time()
        # get the features for the whole dataset
        features = compute_features(device, dataloader, model, len(gtiffdataset))

        if args["verbose"]:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args["verbose"])

        # assign pseudo-labels
        if args["verbose"]:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  gtiffdataset.images)

        sampler = UnifLabelSampler(int(1.0 * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["batch"],
            num_workers=0,
            sampler=sampler,
            pin_memory=True,
        )

        end = time.time()
        loss = train(device, train_dataloader, model, criterion, optimizer, epoch)
        print( clustering_loss, loss)
        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            },
            os.path.join('./', 'checkpoint.pth.tar'))


if __name__=="__main__":
    main()
