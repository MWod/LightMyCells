### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import pathlib

### External Imports ###

### Internal Imports ###

########################


def default_encoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 16, 64, 128, 256]
    output_channels = [16, 64, 128, 256, 512]
    blocks_per_channel = [2, 2, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config

def default_decoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 16, 64, 128, 256]
    output_channels = [16, 64, 128, 256, 512]
    blocks_per_channel = [1, 1, 1, 1, 1]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config




def large_encoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 64, 64, 128, 256, 512]
    output_channels = [64, 64, 128, 256, 512, 512]
    blocks_per_channel = [3, 3, 3, 3, 3, 3]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config

def large_decoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 64, 64, 128, 256, 512]
    output_channels = [64, 64, 128, 256, 512, 512]
    blocks_per_channel = [3, 3, 3, 3, 3, 3]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config





def default_autoencoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 16, 32, 64, 128, 256]
    output_channels = [16, 32, 64, 128, 256, 256]
    blocks_per_channel = [2, 2, 2, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config

def default_autodecoder_config() -> dict:
    ### Define Params ###
    channels = [16, 32, 64, 64, 128, 256]
    blocks_per_channel = [1, 1, 1, 1, 1, 1]
    
    ### Parse ###
    config = {}
    config['channels'] = channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config












def large_autoencoder_config() -> dict:
    ### Define Params ###
    input_channels = [1, 32, 64, 256, 256]
    output_channels = [32, 64, 256, 256, 512]
    blocks_per_channel = [3, 3, 3, 3, 3]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config

def large_autodecoder_config() -> dict:
    ### Define Params ###
    channels = [32, 64, 128, 256, 512]
    blocks_per_channel = [1, 1, 1, 1, 1]
    
    ### Parse ###
    config = {}
    config['channels'] = channels
    config['blocks_per_channel'] = blocks_per_channel
    config['image_size'] = None
    return config












def default_transformer_autoencoder_config() -> dict:
    ### Define Params ###
    in_channels = 1
    img_size = (512, 512)
    patch_size = (16, 16)
    spatial_dims = 2
    hidden_size = 256
    mlp_dim = 1024
    num_layers = 12
    num_heads = 16
    
    ### Parse ###
    config = {}
    config['in_channels'] = in_channels
    config['img_size'] = img_size
    config['patch_size'] = patch_size
    config['spatial_dims'] = spatial_dims
    config['hidden_size'] = hidden_size
    config['mlp_dim'] = mlp_dim
    config['num_layers'] = num_layers
    config['num_heads'] = num_heads
    return config

def default_transformer_autodecoder_config() -> dict:
    ### Define Params ###
    img_size = (512, 512)
    patch_size = (16, 16)
    out_channels = 1
    deconv_chns = 128
    hidden_size = 256
    spatial_dims = 2
    spatial_size = (512, 512)
        
    ### Parse ###
    config = {}
    config['img_size'] = img_size
    config['patch_size'] = patch_size
    config['out_channels'] = out_channels
    config['deconv_chns'] = deconv_chns
    config['hidden_size'] = hidden_size
    config['spatial_dims'] =  spatial_dims
    config['spatial_size'] = spatial_size
    return config
