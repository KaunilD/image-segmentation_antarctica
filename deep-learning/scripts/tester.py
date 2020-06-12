import torch
import cv2
import numpy as np
import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unet import UNet
import torch.nn as nn
import os
import glob

if __name__=="__main__":

    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.])
    ]) 

    image_pth   = "/home/kadh5719/development/git/independent-study/deep-learning/scripts/sample.png"
    image       = cv2.imread(image_pth)
    image_t     = t(image)
    image_t     = torch.unsqueeze(image_t, 0)


    model = UNet()
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)
    model_pths = glob.glob("../models/bedrock/3/unet-*.pth")
    for model_pth in model_pths:
        
        print(model_pth)
        model.load_state_dict(torch.load(model_pth)["model"])
        model.to(device)
        model.eval()

        output = model(image_t)
        output = output.squeeze().detach().cpu().numpy()
        plt.imsave("../models/bedrock/output/{}_{}.png".format(
            os.path.basename(image_pth).split(".")[0], 
            os.path.basename(model_pth).split(".")[0]), 
        output)
        plt.imsave("../models/bedrock/output/{}_{}_binary.png".format(
            os.path.basename(image_pth).split(".")[0], 
            os.path.basename(model_pth).split(".")[0]), 
        output*255)
