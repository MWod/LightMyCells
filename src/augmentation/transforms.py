### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import torchio as tio
import SimpleITK as sitk

### Internal Imports ###
from helpers import utils as u
from paths import paths as p

########################


def geometric_flips():
    flip = tio.RandomFlip(axes=(1, 2), p=0.65)
    return tio.Compose([flip])

def geometric_simple():
    flip = tio.RandomFlip(axes=(1, 2), p=0.5)
    return tio.Compose([flip])

def geometric():
    flip = tio.RandomFlip(axes=(1, 2), p=0.75)
    affine = tio.RandomAffine(scales=(1.0, 1.0, 0.8, 2.0, 0.8, 2.0), degrees=(-45, 45, 0.0, 0.0, 0.0, 0.0), translation=(0, 0, 0, 0, 0, 0), p=0.75)
    elastic = tio.RandomElasticDeformation(num_control_points=15, max_displacement=(0, 30, 30), p=0.75)
    return tio.Compose([flip, affine, elastic])

def intensity_simple():
    blur = tio.RandomBlur(std=(0, 1.0), p=0.5)
    noise = tio.RandomNoise(mean=0, std =(0, 0.5), p=0.5)
    gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    return tio.Compose([blur, noise, gamma])