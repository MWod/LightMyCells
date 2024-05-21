### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib
import xml.etree.ElementTree as ET
import math

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import torch
import torch.nn.functional as F

### Internal Imports ###

########################



def mean_absolute_error(tensor_1, tensor_2):
    return tc.mean(tc.abs(tensor_1 - tensor_2))

def mean_squared_error(tensor_1, tensor_2):
    return tc.mean((tensor_1 - tensor_2)**2)

def euclidean_distance(tensor_1, tensor_2):
    return tc.cdist(tensor_1.view(tensor_1.shape[0], -1), tensor_2.view(tensor_1.shape[0], -1))

def cosine_distance(tensor_1, tensor_2):
    return -tc.mean(tc.cosine_similarity(tensor_1.view(tensor_1.shape[0], -1), tensor_2.view(tensor_1.shape[0], -1)))

def pearson_correlation_coefficient(tensor_1, tensor_2):
    t1_mean = tc.mean(tensor_1)
    t2_mean = tc.mean(tensor_2)
    numerator = tc.sum((tensor_1 - t1_mean)*(tensor_2 - t2_mean))
    denominator = tc.linalg.norm(tensor_1 - t1_mean) * tc.linalg.norm(tensor_2 - t2_mean)
    return -numerator / (denominator + 1e-5)

def normalized_cross_correlation(tensor_1, tensor_2):
    return ncc_local_tc(tensor_1, tensor_2)

def structural_similarity_index_measure(tensor_1, tensor_2):
    return -ssim(tensor_1, tensor_2)





























def ncc_local_tc(sources: tc.Tensor, targets: tc.Tensor, **params):
    """
    Local normalized cross-correlation (as cost function) using PyTorch tensors.

    Implementation inspired by VoxelMorph (with some modifications).

    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 5
    window = (win_size, ) * ndim
    sum_filt = tc.ones([1, 1, *window], device=sources.device, dtype=sources.dtype)
    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)


def gaussian(window_size, sigma, device):
    gauss = tc.tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device)
    return gauss/gauss.sum()

def create_window(window_size, channel, device):
    _1D_window = gaussian(window_size, 1.5, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = tc.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()).to(device)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, device=img1.device)
    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    window = window.to(img1.device).to(img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, size_average)