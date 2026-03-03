import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward
import cv2
from torchvision import transforms
  
class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d( out_ch),
            )
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.conv1x1(identity)

        x = self.relu(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        n_channels,
        number
    ):
        super().__init__()

        self.number = number
       
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)

        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def denormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        return ((tensor * std + mean) * 255).clamp(0, 255).byte()

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    
    def forward(self, x):
        
        # block0
        x = self.conv00(x)

        # block1
        x = self.maxpooling(x)
        x = self.conv20(x)

        # block2
        x = self.maxpooling(x)
        x = self.conv40(x)

        # block3
        x = self.maxpooling(x)
        x = self.conv60(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x