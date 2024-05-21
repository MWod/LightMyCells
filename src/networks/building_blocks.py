### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import math
import torch as tc
import torch.nn.functional as F

### Internal Imports ###


########################

def pad(image : tc.Tensor, template : tc.Tensor) -> tc.Tensor:
    pad_x = math.fabs(image.size(3) - template.size(3))
    pad_y = math.fabs(image.size(2) - template.size(2))
    b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
    b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
    image = F.pad(image, (b_x, e_x, b_y, e_y))
    return image

def resample(x : tc.Tensor, template : tc.Tensor) -> tc.Tensor:
    return F.interpolate(x, template.shape[2:], mode='biilinear')

class ConvolutionalBlock(tc.nn.Module):
    def __init__(self, input_size : int, output_size : int, leaky_alpha : float=0.01):
        super(ConvolutionalBlock, self).__init__()

        self.module = tc.nn.Sequential(
            tc.nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),
            tc.nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

    def forward(self, x : tc.Tensor):
        return self.module(x)
    

class ResidualBlock(tc.nn.Module):
    def __init__(self, input_size : int, output_size : int, leaky_alpha : float=0.01):
        super(ResidualBlock, self).__init__()

        self.module = tc.nn.Sequential(
            tc.nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),
            tc.nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

        self.conv = tc.nn.Sequential(
            tc.nn.Conv2d(input_size, output_size, 1)
        )

    def forward(self, x : tc.Tensor):
        return self.module(x) + self.conv(x)
    
    

class ResidualBlockIN(tc.nn.Module):
    def __init__(self, input_size : int, output_size : int, leaky_alpha : float=0.01):
        super(ResidualBlockIN, self).__init__()

        self.module = tc.nn.Sequential(
            tc.nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            tc.nn.InstanceNorm2d(output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),
            tc.nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            tc.nn.InstanceNorm2d(output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

        self.conv = tc.nn.Sequential(
            tc.nn.Conv2d(input_size, output_size, 1)
        )

    def forward(self, x : tc.Tensor):
        return self.module(x) + self.conv(x)