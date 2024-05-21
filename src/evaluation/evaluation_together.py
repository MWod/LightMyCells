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
from networks import runet, autoencoder
from networks import configs as cfg
from helpers import objective_functions as of


########################

def percentile_normalization_no_scale(image, pmin=2, pmax=99.8, axis=None, dtype=np.uint16):
    '''
    Compute a percentile normalization for the given image.

    Parameters:
    - image (array): array of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed. 
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    '''

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_p  = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        # print(f"Same min {low_p} and high {high_p}, image may be empty")
    else:
        img_norm = (image - low_p) / ( high_p - low_p )
        img_norm = img_norm
    return img_norm

def normalization(tensor):
    return (tensor - tc.min(tensor)) / (tc.max(tensor) - tc.min(tensor))

def preprocess_image(image):
    image = image.astype(np.float32)
    tensor = tc.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = normalization(tensor)
    return tensor

def patch_based_inference(tensor, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin):
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
        actin = decoder_actin(embedding)
        mitochondria = decoder_mitochondria(embedding)
        nucleus = decoder_nucleus(embedding)
        tubulin = decoder_tubulin(embedding)
        # outputs = objective
        # outputs = outputs.unsqueeze(0).permute(1, 2, 0, 3, 4)
        outputs = tc.cat((nucleus, mitochondria, tubulin, actin), dim=1)
        outputs = outputs.unsqueeze(0).permute(1, 2, 0, 3, 4)
        aggregator.add_batch(outputs, locations)
        
    output_tensor = aggregator.get_output_tensor().to(tc.float32).permute(1, 0, 2, 3).to(tensor.device)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Output min/max: {output_tensor.min(), output_tensor.max()}")
    actin = output_tensor[:, 0:1, :, :]
    mitochondria = output_tensor[:, 1:2, :, :]
    nucleus = output_tensor[:, 2:3, :, :]
    tubulin = output_tensor[:, 3:4, :, :]
    return actin, mitochondria, nucleus, tubulin

def postprocess_output(actin, mitochondria, nucleus, tubulin):
    actin = actin.detach().cpu().numpy()[0, 0]
    print(f"Postprocessed actin min/max: {actin.min(), actin.max()}")
    mitochondria = mitochondria.detach().cpu().numpy()[0, 0]
    print(f"Postprocessed mitochondria min/max: {mitochondria.min(), mitochondria.max()}")
    nucleus = nucleus.detach().cpu().numpy()[0, 0]
    print(f"Postprocessed nucleus min/max: {nucleus.min(), nucleus.max()}")
    tubulin = tubulin.detach().cpu().numpy()[0, 0]
    print(f"Postprocessed tubulin min/max: {tubulin.min(), tubulin.max()}")
    return actin, mitochondria, nucleus, tubulin


def get_model(device="cpu"):
    checkpoint_path = p.checkpoints_path / "LightMyCells_Baseline_3" / 'epoch=76_loss.ckpt'
    checkpoint = tc.load(checkpoint_path, map_location=tc.device('cpu'))
    state_dict = checkpoint['state_dict']

    encoder_state_dict = {}
    decoder_actin_state_dict = {}
    decoder_mitochondria_state_dict = {}
    decoder_nucleus_state_dict = {}
    decoder_tubulin_state_dict = {}

    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "encoder" in key:
            encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]     
        if "actin_decoder" in key:
            decoder_actin_state_dict[key.replace("actin_decoder.", "")] = state_dict[key]
        if "mitochondria_decoder" in key:
            decoder_mitochondria_state_dict[key.replace("mitochondria_decoder.", "")] = state_dict[key]         
        if "nucleus_decoder" in key:
            decoder_nucleus_state_dict[key.replace("nucleus_decoder.", "")] = state_dict[key]
        if "tubulin_decoder" in key:
            decoder_tubulin_state_dict[key.replace("tubulin_decoder.", "")] = state_dict[key]              
    encoder = autoencoder.Encoder(**cfg.default_autoencoder_config())
    decoder_actin = autoencoder.Decoder(**cfg.default_autodecoder_config())
    decoder_mitochondria = autoencoder.Decoder(**cfg.default_autodecoder_config())
    decoder_nucleus = autoencoder.Decoder(**cfg.default_autodecoder_config())
    decoder_tubulin = autoencoder.Decoder(**cfg.default_autodecoder_config())

    encoder.load_state_dict(encoder_state_dict)
    decoder_actin.load_state_dict(decoder_actin_state_dict)
    decoder_mitochondria.load_state_dict(decoder_mitochondria_state_dict)
    decoder_nucleus.load_state_dict(decoder_nucleus_state_dict)
    decoder_tubulin.load_state_dict(decoder_tubulin_state_dict)

    encoder = encoder.to(device)
    decoder_actin = decoder_actin.to(device)
    decoder_mitochondria = decoder_mitochondria.to(device)
    decoder_nucleus = decoder_nucleus.to(device)
    decoder_tubulin = decoder_tubulin.to(device)

    encoder.eval()
    decoder_actin.eval()
    decoder_mitochondria.eval()
    decoder_nucleus.eval()
    decoder_tubulin.eval()

    return encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin


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


def predict(image, metadata, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin, device="cpu"):
    with tc.no_grad():
        tensor = preprocess_image(image).to(device)
        print(f"Input tensor shape: {tensor.shape}")
        print(f"Tensor Min/Max: {tensor.min(), tensor.max()}")
        actin, mitochondria, nucleus, tubulin = patch_based_inference(tensor, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin)
        actin, mitochondria, nucleus, tubulin = postprocess_output(actin, mitochondria, nucleus, tubulin)
        return actin, mitochondria, nucleus, tubulin
    
    
def run_evaluation(data_path, csv_path, output_folder, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin, device, save_step=20):
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
            nucleus, mitochondria, tubulin, actin = predict(image, metadata, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin, device)

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
                ground_truth = percentile_normalization_no_scale(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "nucleus_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(nucleus, ground_truth)
                print(f"Nucleus MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                nucleus_results.append(to_append)
            
            if type(mitochondria_path) is str:
                ground_truth, _ = read_image(data_path / mitochondria_path)
                ground_truth = percentile_normalization_no_scale(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "mitochondria_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(mitochondria, ground_truth)
                print(f"Mitochondria MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                mitochondria_results.append(to_append)
            
            if type(tubulin_path) is str:
                ground_truth, _ = read_image(data_path / tubulin_path) 
                ground_truth = percentile_normalization_no_scale(ground_truth)
                if i % save_step == 0:
                    save_image(output_folder / str(i) / "tubulin_gt.tiff", ground_truth, metadata)
                mae, mse, cd, pcc, ssim, ed = calculate_losses(tubulin, ground_truth)
                print(f"Tubulin MAE, MSE, CD, PC, SSIM, ED: {mae, mse, cd, pcc, ssim, ed}")
                to_append = (i, input_path, mae, mse, cd, pcc, ssim, ed)
                tubulin_results.append(to_append)
            
            if type(actin_path) is str:
                ground_truth, _ = read_image(data_path / actin_path) 
                ground_truth = percentile_normalization_no_scale(ground_truth)
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
    device = "cuda:1"
    data_path = p.raw_data_path / "RAW"
    csv_path = p.data_path / "validation_dataset.csv"
    output_folder = p.results_path / "Test_AllTogether"
    encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin = get_model(device)
    run_evaluation(data_path, csv_path, output_folder, encoder, decoder_actin, decoder_mitochondria, decoder_nucleus, decoder_tubulin, device)



if __name__ == "__main__":
    run()