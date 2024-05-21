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










class Encoder(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int], image_size : tuple):
        super(Encoder, self).__init__()
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
        cx = x
        for i in range(self.num_channels):
            cx = getattr(self, f"encoder_{i}")(cx)
        return cx

class Decoder(tc.nn.Module):
    def __init__(self, channels : Iterable[int],  blocks_per_channel : Iterable[int], image_size : tuple):
        super(Decoder, self).__init__()
        self.channels = channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(channels)
        self.image_size = image_size
        
        for i, (oc, bpc) in enumerate(zip(channels, blocks_per_channel)):
            module_list = []
            coc = oc if i == self.num_channels - 1 else self.channels[i+1]
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
        self.last_layer = tc.nn.Conv2d(self.channels[0], 1, 1)    

    def forward(self, embeddings : Iterable[tc.Tensor]) -> tc.Tensor:
        cx = embeddings
        for i in range(self.num_channels - 1, -1, -1):
            cx = getattr(self, f"decoder_{i}")(cx)    
        cx = self.last_layer(cx)    
        _, _, d, h = cx.shape        
        if self.image_size is not None and (d, h) != (self.image_size[0], self.image_size[1]):
            result = F.interpolate(result, (d, h), mode='biilinear') 
        else:
            result = cx
        return result
        
        
        
        
        
        
class EncoderIN(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int], image_size : tuple):
        super(EncoderIN, self).__init__()
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
                    module_list.append(bb.ResidualBlockIN(ic, oc))
                else:
                    module_list.append(bb.ResidualBlockIN(oc, oc))
            cic = ic if bpc == 0 else oc
            module_list.append(tc.nn.Conv2d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.InstanceNorm2d(oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"encoder_{i}", layer)
        
    def forward(self, x : tc.Tensor) -> Iterable[tc.Tensor]:
        _, _, d, h = x.shape
        if self.image_size is not None and (d, h) != (self.image_size[0], self.image_size[1]):
            x = F.interpolate(x, self.image_size, mode='biilinear')
        cx = x
        for i in range(self.num_channels):
            cx = getattr(self, f"encoder_{i}")(cx)
        return cx

class DecoderIN(tc.nn.Module):
    def __init__(self, channels : Iterable[int],  blocks_per_channel : Iterable[int], image_size : tuple):
        super(DecoderIN, self).__init__()
        self.channels = channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(channels)
        self.image_size = image_size
        
        for i, (oc, bpc) in enumerate(zip(channels, blocks_per_channel)):
            module_list = []
            coc = oc if i == self.num_channels - 1 else self.channels[i+1]
            for j in range(bpc):
                if j == 0: 
                    module_list.append(bb.ResidualBlockIN(coc, oc))
                else:
                    module_list.append(bb.ResidualBlockIN(oc, oc))
            cic = coc if bpc == 0 else oc
            module_list.append(tc.nn.ConvTranspose2d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.InstanceNorm2d(oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"decoder_{i}", layer)   
        self.last_layer = tc.nn.Conv2d(self.channels[0], 1, 1)    

    def forward(self, embeddings : Iterable[tc.Tensor]) -> tc.Tensor:
        cx = embeddings
        for i in range(self.num_channels - 1, -1, -1):
            cx = getattr(self, f"decoder_{i}")(cx)    
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
    encoder_config = cfg.default_autoencoder_config()
    decoder_config = cfg.default_autodecoder_config()
    encoder = Encoder(**encoder_config).to(device)
    decoder_1 = Decoder(**decoder_config).to(device)
    decoder_2 = Decoder(**decoder_config).to(device)
    decoder_3 = Decoder(**decoder_config).to(device)
    decoder_4 = Decoder(**decoder_config).to(device)
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
    print(f"Embedding shape: {embedding.shape}")
    print(f"Result 1 shape: {result_1.shape}")
    print(f"Result 2 shape: {result_2.shape}")
    print(f"Result 3 shape: {result_3.shape}")
    print(f"Result 4 shape: {result_4.shape}")
    ts.summary(encoder, input_data=input, device=device)  
    ts.summary(decoder_1, input_data=embedding, device=device)  
    
    
def test_channels_2():
    device = "cuda:1"
    encoder_config = cfg.large_autoencoder_config()
    decoder_config = cfg.large_autodecoder_config()
    encoder = Encoder(**encoder_config).to(device)
    decoder_1 = Decoder(**decoder_config).to(device)
    decoder_2 = Decoder(**decoder_config).to(device)
    decoder_3 = Decoder(**decoder_config).to(device)
    decoder_4 = Decoder(**decoder_config).to(device)
    num_samples = 4
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
    test_channels_2()
    pass

if __name__ == "__main__":
    run()