### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Sequence
import pathlib
import math

### External Imports ###
import torch as tc
import torch.nn.functional as F


from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import deprecated_arg, ensure_tuple_rep, is_sqrt

### Internal Imports ###

import building_blocks as bb
import configs as cfg
import torchsummary as ts

########################




class Encoder(tc.nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()
        if not is_sqrt(patch_size):
            raise ValueError(f"patch_size should be square number, got {patch_size}.")
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.spatial_dims = spatial_dims
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(f"patch_size={patch_size} should be divisible by img_size={img_size}.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.blocks = tc.nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = tc.nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x




class Decoder(tc.nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        spatial_dims: int = 2,
        spatial_size: tuple = (512, 512),
    ) -> None:
        super().__init__()
        if not is_sqrt(patch_size):
            raise ValueError(f"patch_size should be square number, got {patch_size}.")
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.spatial_dims = spatial_dims
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(f"patch_size={patch_size} should be divisible by img_size={img_size}.")

        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        up_kernel_size = [int(math.sqrt(i)) for i in self.patch_size]
        self.conv3d_transpose_1 = conv_trans(hidden_size, deconv_chns, kernel_size=up_kernel_size, stride=up_kernel_size)
        self.conv3d_transpose_2 = conv_trans(in_channels=deconv_chns, out_channels=out_channels, kernel_size=up_kernel_size, stride=up_kernel_size)
        self.spatial_size = spatial_size


    def forward(self, x):
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(self.spatial_size, self.patch_size)]
        x = tc.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose_1(x)
        x = self.conv3d_transpose_2(x)
        return x












### Verification ###

def test_channels_1():
    device = "cpu"
    encoder = Encoder(**cfg.default_transformer_autoencoder_config()).to(device)
    decoder_1 = Decoder(**cfg.default_transformer_autodecoder_config()).to(device)
    decoder_2 = Decoder(**cfg.default_transformer_autodecoder_config()).to(device)
    decoder_3 = Decoder(**cfg.default_transformer_autodecoder_config()).to(device)
    decoder_4 = Decoder(**cfg.default_transformer_autodecoder_config()).to(device)
    num_samples = 16
    num_channels = 1
    y_size = 512
    x_size = 512
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    embedding = encoder(input)
    result_1 = decoder_1(embedding)
    result_2 = decoder_2(embedding)
    result_3 = decoder_3(embedding)
    result_4 = decoder_4(embedding)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Result 1 shape: {result_1.shape}")
    print(f"Result 2 shape: {result_2.shape}")
    print(f"Result 3 shape: {result_3.shape}")
    print(f"Result 4 shape: {result_4.shape}")
    ts.summary(encoder, input_data=input, device=device)  
    ts.summary(decoder_1, input_data=embedding, device=device)  
    
    

def run():
    test_channels_1()
    pass

if __name__ == "__main__":
    run()