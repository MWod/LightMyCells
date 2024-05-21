### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import pathlib

### External Imports ###
import torch as tc
import torch.nn.functional as F

### Internal Imports ###

import building_blocks as bb
import configs as cfg
import torchsummary as ts

########################







class RUNetEncoder(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int], image_size : tuple):
        super(RUNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        self.image_size = image_size
        
        if len(self.input_channels) != len(self.output_channels):
            raise ValueError("Number of input channels must be equal to the number of output channels.")
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            for j in range(bpc):
                if j == 0:
                    module_list.append(bb.ResidualBlock(ic, oc))
                else:
                    module_list.append(bb.ResidualBlock(oc, oc))
            cic = ic if bpc == 0 else oc
            module_list.append(tc.nn.Conv2d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.GroupNorm(oc, oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"encoder_{i}", layer)
        
    def forward(self, x : tc.Tensor) -> Iterable[tc.Tensor]:
        _, _, d, h = x.shape
        if self.image_size is not None and (d, h) != (self.image_size[0], self.image_size[1]):
            x = F.interpolate(x, self.image_size, mode='biilinear')
            
        embeddings = []
        cx = x
        for i in range(self.num_channels):
            cx = getattr(self, f"encoder_{i}")(cx)
            embeddings.append(cx)
        return embeddings

class RUNetDecoder(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int], image_size : tuple):
        super(RUNetDecoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        self.image_size = image_size
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            coc = oc if i == self.num_channels - 1 else oc + output_channels[i + 1]
            for j in range(bpc):
                if j == 0: 
                    module_list.append(bb.ResidualBlock(coc, oc))
                else:
                    module_list.append(bb.ResidualBlock(oc, oc))
            cic = coc if bpc == 0 else oc
            module_list.append(tc.nn.ConvTranspose2d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.GroupNorm(oc, oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"decoder_{i}", layer)   
        self.last_layer = tc.nn.Conv2d(self.output_channels[0], 1, 1)    

    def forward(self, embeddings : Iterable[tc.Tensor]) -> tc.Tensor:
        for i in range(self.num_channels - 1, -1, -1):
            if i == self.num_channels - 1:
                cx = getattr(self, f"decoder_{i}")(embeddings[i])         
            else:
                cx = getattr(self, f"decoder_{i}")(tc.cat((bb.pad(cx, embeddings[i]), embeddings[i]), dim=1))
        cx = self.last_layer(cx)    
        _, _, d, h = cx.shape        
        if self.image_size is not None and (d, h) != (self.image_size[0], self.image_size[1]):
            result = F.interpolate(result, (d, h), mode='biilinear') 
        else:
            result = cx
        return result
        

### Verification ###

def test_channels_1():
    device = "cuda:1"
    encoder_config = cfg.default_encoder_config()
    decoder_config = cfg.default_decoder_config()
    encoder = RUNetEncoder(**encoder_config).to(device)
    decoder_1 = RUNetDecoder(**decoder_config).to(device)
    decoder_2 = RUNetDecoder(**decoder_config).to(device)
    decoder_3 = RUNetDecoder(**decoder_config).to(device)
    decoder_4 = RUNetDecoder(**decoder_config).to(device)
    num_samples = 16
    num_channels = 1
    y_size = 256
    x_size = 256
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    embedding = encoder(input)
    result_1 = decoder_1(embedding)
    result_2 = decoder_2(embedding)
    result_3 = decoder_3(embedding)
    result_4 = decoder_4(embedding)
    print(f"Len embeddign: {len(embedding)}")
    print(f"Embedding 1 shape: {embedding[0].shape}")
    print(f"Embedding 2 shape: {embedding[1].shape}")
    print(f"Embedding 3 shape: {embedding[2].shape}")
    print(f"Embedding 4 shape: {embedding[3].shape}")
    print(f"Embedding 5 shape: {embedding[4].shape}")
    print(f"Result 1 shape: {result_1.shape}")
    print(f"Result 2 shape: {result_2.shape}")
    print(f"Result 3 shape: {result_3.shape}")
    print(f"Result 4 shape: {result_4.shape}")
    ts.summary(encoder, input_data=input, device=device)  
    
    
def test_channels_2():
    device = "cuda:1"
    encoder_config = cfg.large_encoder_config()
    decoder_config = cfg.large_decoder_config()
    encoder = RUNetEncoder(**encoder_config).to(device)
    decoder_1 = RUNetDecoder(**decoder_config).to(device)
    num_samples = 16
    num_channels = 1
    y_size = 512
    x_size = 512
    input = tc.randn((num_samples, num_channels, y_size, x_size), device=device)
    embedding = encoder(input)
    result_1 = decoder_1(embedding)
    print(f"Len embeddign: {len(embedding)}")
    print(f"Embedding 1 shape: {embedding[0].shape}")
    print(f"Embedding 2 shape: {embedding[1].shape}")
    print(f"Embedding 3 shape: {embedding[2].shape}")
    print(f"Embedding 4 shape: {embedding[3].shape}")
    print(f"Embedding 5 shape: {embedding[4].shape}")
    print(f"Embedding 6 shape: {embedding[5].shape}")
    print(f"Result 1 shape: {result_1.shape}")
    ts.summary(encoder, input_data=input, device=device)  


def run():
    test_channels_2()
    pass

if __name__ == "__main__":
    run()