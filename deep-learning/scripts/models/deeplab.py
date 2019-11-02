import torch
import torch.nn as nn
import torch.nn.functional as F
from . import aspp
from . import decoder
from . import resnet

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False, pretrained=False):
        super(DeepLab, self).__init__()
        self.name = "deeplab"

        self.backbone = resnet.ResNet(
            resnet.Bottleneck,
            [3, 4, 23, 3],
            output_stride,
            nn.BatchNorm2d,
            pretrained=pretrained
        )
        self.aspp = aspp.ASPP(self.backbone, output_stride, nn.BatchNorm2d)
        self.decoder = decoder.Decoder(num_classes, 'resnet', nn.BatchNorm2d)
        self.softmax = nn.Softmax(dim=1) # should return [b,c=3,h,w], normalized over, c dimension

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return self.softmax(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
