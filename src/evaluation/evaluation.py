### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib
import xml.etree.ElementTree as ET

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
# from aicsimageio.readers.bioformats_reader import BioFile
import tifffile
import torchio as tio
import xmltodict

### Internal Imports ###

from paths import paths as p
from datasets import dataset_direct_mha
from networks import runet
from networks import configs as cfg
from helpers import objective_functions as of


########################


def normalization(tensor):
    return (tensor - tc.min(tensor)) / (tc.max(tensor) - tc.min(tensor))

def preprocess_image(image):
    image = image.astype(np.float32)
    tensor = tc.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = normalization(tensor)
    return tensor

def patch_based_inference(tensor, encoder, decoder):
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    current_input = tensor
    print(f"Current input shape: {current_input.shape}")
    current_subject = tio.Subject(input = tio.ScalarImage(tensor=current_input))
    grid_sampler = tio.inference.GridSampler(
        current_subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=32)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
    for patches_batch in patch_loader:
        input_tensor = patches_batch['input'][tio.DATA].to(tensor.device)
        locations = patches_batch[tio.LOCATION]
        embedding = encoder(input_tensor[:, 0, :, :, :])
        objective = decoder(embedding)
        outputs = objective
        outputs = outputs.unsqueeze(0).permute(1, 2, 0, 3, 4)
        aggregator.add_batch(outputs, locations)
        
    output_tensor = aggregator.get_output_tensor().to(tc.float32).permute(1, 0, 2, 3).to(tensor.device)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Output min/max: {output_tensor.min(), output_tensor.max()}")
    return output_tensor

def postprocess_output(output):
    # output = normalization(output)
    array = output.detach().cpu().numpy()[0, 0]
    # array = percentile_normalization(array, dtype=np.uint16)
    # array = (array * np.iinfo(np.uint16).max).astype(np.uint16)
    print(f"Postprocessed min/max: {array.min(), array.max()}")
    return array


def get_actin_model(device="cpu"):
    # checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_1" / 'epoch=47_loss.ckpt'
    checkpoint_path = p.project_path / "Files" / "Actin_Sampling.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {}
    decoder_state_dict = {}
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "encoder" in key:
            encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]         
        if "decoder" in key:
            decoder_state_dict[key.replace("decoder.", "")] = state_dict[key]          
    encoder_config = cfg.default_encoder_config()
    decoder_config = cfg.default_decoder_config()
    encoder = runet.RUNetEncoder(**encoder_config)
    decoder = runet.RUNetDecoder(**decoder_config)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def get_tubulin_model(device="cpu"):
    # checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_1" / 'epoch=71_loss.ckpt'
    checkpoint_path = p.project_path / "Files" / "Tubulin_Sampling.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {}
    decoder_state_dict = {}
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "encoder" in key:
            encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]         
        if "decoder" in key:
            decoder_state_dict[key.replace("decoder.", "")] = state_dict[key]          
    encoder_config = cfg.default_encoder_config()
    decoder_config = cfg.default_decoder_config()
    encoder = runet.RUNetEncoder(**encoder_config)
    decoder = runet.RUNetDecoder(**decoder_config)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def get_nucleus_model(device="cpu"):
    # checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_1" / 'epoch=26_loss.ckpt'
    checkpoint_path = p.project_path / "Files" / "Nucleus_Sampling.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {}
    decoder_state_dict = {}
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "encoder" in key:
            encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]         
        if "decoder" in key:
            decoder_state_dict[key.replace("decoder.", "")] = state_dict[key]          
    encoder_config = cfg.default_encoder_config()
    decoder_config = cfg.default_decoder_config()
    encoder = runet.RUNetEncoder(**encoder_config)
    decoder = runet.RUNetDecoder(**decoder_config)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def get_mitochondria_model(device="cpu"):
    # checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_1" / 'epoch=28_loss.ckpt'
    checkpoint_path = p.project_path / "Files" / "Mitochondria_Sampling.ckpt"
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {}
    decoder_state_dict = {}
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "encoder" in key:
            encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]         
        if "decoder" in key:
            decoder_state_dict[key.replace("decoder.", "")] = state_dict[key]          
    encoder_config = cfg.default_encoder_config()
    decoder_config = cfg.default_decoder_config()
    encoder = runet.RUNetEncoder(**encoder_config)
    decoder = runet.RUNetDecoder(**decoder_config)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def read_image(location):
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray()
        metadata = tif.ome_metadata
    return image_data, metadata

def save_image(location, array, metadata):
    pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
    physical_size_x = float(pixels["@PhysicalSizeX"])
    physical_size_y = float(pixels["@PhysicalSizeY"])
    tifffile.imwrite(location,
                     array,
                     description=metadata,
                     resolution=(physical_size_x, physical_size_y),
                     metadata=pixels,
                     tile=(128, 128),
                     )


def predict(image, metadata, encoder, decoder, device="cpu"):
    with tc.no_grad():
        tensor = preprocess_image(image).to(device)
        print(f"Input tensor shape: {tensor.shape}")
        print(f"Tensor Min/Max: {tensor.min(), tensor.max()}")
        output = patch_based_inference(tensor, encoder, decoder)
        output = postprocess_output(output)
        return output
    
    
def run_evaluation(data_path, csv_path, output_folder, nucleus_model, mitochondria_model, actin_model, tubulin_model, device, save_step=100):
    dataframe = pd.read_csv(csv_path)
    print(f"Number of cases: {len(dataframe)}")
    
    nucleus_results = []
    mitochondria_results = []
    tubulin_results = []
    actin_results = []
    
    for i in range(len(dataframe)):
        with tc.no_grad():
            print(f"Current case: {i+1} / {len(dataframe)}")
            current_case = dataframe.iloc[i]
            input_path = current_case['Input Path']
            nucleus_path = current_case['Nucleus Path']
            mitochondria_path = current_case['Mitochondria Path']
            tubulin_path = current_case['Tubulin Path']
            actin_path = current_case['Actin Path']
            
            
            image, metadata = read_image(data_path / input_path)
            print(f"Image shape: {image.shape}")
            nucleus = predict(image, metadata, nucleus_model[0], nucleus_model[1], device)
            print(f"Nucleus shape: {nucleus.shape}")
            mitochondria = predict(image, metadata, mitochondria_model[0], mitochondria_model[1], device)
            print(f"Mitochondria shape: {mitochondria.shape}")
            tubulin = predict(image, metadata, tubulin_model[0], tubulin_model[1], device)
            print(f"Tubulin shape: {tubulin.shape}")
            actin = predict(image, metadata, actin_model[0], actin_model[1], device)
            print(f"Actin shape: {actin.shape}")
            
            if i % save_step == 0:
                if not os.path.isdir(output_folder / str(i)):
                    os.makedirs(output_folder / str(i))
                save_image(output_folder / str(i) / "image.tiff", image, metadata)
                save_image(output_folder / str(i) / "nucleus.tiff", nucleus, metadata)
                save_image(output_folder / str(i) / "mitochondria.tiff", mitochondria, metadata)
                save_image(output_folder / str(i) / "tubulin.tiff", tubulin, metadata)
                save_image(output_folder / str(i) / "actin.tiff", actin, metadata)
            
            if type(nucleus_path) is str:
                ground_truth, _ = read_image(data_path / nucleus_path) 
                ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) 
                # ground_truth = percentile_normalization(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "nucleus_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(nucleus, ground_truth)
                print(f"Nucleus MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                nucleus_results.append(to_append)
            
            if type(mitochondria_path) is str:
                ground_truth, _ = read_image(data_path / mitochondria_path)
                ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) 
                # ground_truth = percentile_normalization(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "mitochondria_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(mitochondria, ground_truth)
                print(f"Mitochondria MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                mitochondria_results.append(to_append)
            
            if type(tubulin_path) is str:
                ground_truth, _ = read_image(data_path / tubulin_path) 
                ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) 
                # ground_truth = percentile_normalization(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "tubulin_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(tubulin, ground_truth)
                print(f"Tubulin MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                tubulin_results.append(to_append)
            
            if type(actin_path) is str:
                ground_truth, _ = read_image(data_path / actin_path) 
                ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) 
                # ground_truth = percentile_normalization(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "actin_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(actin, ground_truth)
                print(f"Actin MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                actin_results.append(to_append)
            
    nucleus_dataframe = pd.DataFrame(nucleus_results, columns=['ID', 'Input Path', 'MAE', 'MSE', 'CD', 'PCC', 'SSIM', 'ED'])
    nucleus_dataframe.to_csv(output_folder / "nucleus.csv", index=False)
    
    mitochondria_dataframe = pd.DataFrame(mitochondria_results, columns=['ID', 'Input Path', 'MAE', 'MSE', 'CD', 'PCC', 'SSIM', 'ED'])
    mitochondria_dataframe.to_csv(output_folder / "mitochondria.csv", index=False)
    
    tubulin_dataframe = pd.DataFrame(tubulin_results, columns=['ID', 'Input Path', 'MAE', 'MSE', 'CD', 'PCC', 'SSIM', 'ED'])
    tubulin_dataframe.to_csv(output_folder / "tubulin.csv", index=False)
    
    actin_dataframe = pd.DataFrame(actin_results, columns=['ID', 'Input Path', 'MAE', 'MSE', 'CD', 'PCC', 'SSIM', 'ED'])
    actin_dataframe.to_csv(output_folder / "actin.csv", index=False)
            


def calculate_losses(output, ground_truth):
    output = tc.from_numpy(output.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    ground_truth = tc.from_numpy(ground_truth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mae = of.mean_absolute_error(output, ground_truth).item()
    mse = of.mean_squared_error(output, ground_truth).item()
    cd = -of.cosine_distance(output, ground_truth).item()
    pcc = -of.pearson_correlation_coefficient(output, ground_truth).item()
    ssim = -of.structural_similarity_index_measure(normalization(output), normalization(ground_truth)).item()
    ed = of.euclidean_distance(output, ground_truth).item()
    return mae, mse, cd, pcc, ssim, ed



def run():
    pass


if __name__ == "__main__":
    run()