### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib
import random

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






def percentile_normalization(image, pmin=2, pmax=99.8, axis=None, dtype=np.uint16):
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
        dtype_max = np.iinfo(dtype).max
        img_norm = dtype_max * (image - low_p) / ( high_p - low_p )
        img_norm = img_norm.astype(dtype)
    return img_norm


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











class TrainingDataset(tc.utils.data.Dataset):
    """
    TODO
    """
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        iteration_size : int = -1,
        transforms : Callable = None,
        return_metadata : bool = False,
        return_input_path : bool = False,
        use_sampling : bool = False,
        nucleus_weight : float = 1,
        mitochondria_weight : float = 1,
        tubulin_weight : float = 1,
        actin_weight : float = 1,
        patch_size = (512, 512),
        num_patches = 4):
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.iteration_size = iteration_size
        self.transforms = transforms
        self.return_metadata = return_metadata
        self.return_input_path = return_input_path
        self.use_sampling = use_sampling
        self.dataframe = pd.read_csv(self.csv_path)
        self.patch_size = patch_size
        self.num_patches = num_patches
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)
            
        self.ids = np.arange(len(self.dataframe))
        self.nucleus_weight = nucleus_weight
        self.mitochondria_weight = mitochondria_weight
        self.tubulin_weight = tubulin_weight
        self.actin_weight = actin_weight
        self.weights = self.get_weights()

    def get_weights(self):
        weights = []
        for idx in range(len(self.dataframe)):
            current_case = self.dataframe.loc[idx]
            weight = 0.0
            # Nucleus
            nucleus_path = current_case['Nucleus Path']
            if type(nucleus_path) is str:
                weight += self.nucleus_weight
            # Mitochondria
            mitochondria_path = current_case['Mitochondria Path']
            if type(mitochondria_path) is str:
                weight += self.mitochondria_weight
            # Tubulin
            tubulin_path = current_case['Tubulin Path']
            if type(tubulin_path) is str:
                weight += self.tubulin_weight
            # Actin
            actin_path = current_case['Actin Path']
            if type(actin_path) is str:
                weight += self.actin_weight
        
            weights.append(weight)
            
        weights = np.array(weights)
        print(f"Weights shape: {weights.shape}")
        print(f"Weights sum before norm: {np.sum(weights)}")
        norm = np.linalg.norm(weights, ord=1)
        print(f"Weights norm: {norm}")
        weights = weights / norm
        print(f"Weights sum after norm: {np.sum(weights)}")
        return weights

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def parse_metadata(self, image):
        metadata = {}
        # TODO
        return metadata

    def load_ground_truth(self, array, relative_path):
        if type(relative_path) is float:
            ground_truth = np.zeros_like(array, dtype=np.uint16)
            available = False
        else:
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path / relative_path)).astype(np.uint16)[np.newaxis, np.newaxis, np.newaxis, :, :]
            available = True
        return ground_truth, available
    
    def sample_idx(self):
        idx = np.random.choice(self.ids, size=1, p=self.weights)[0]
        return idx
        
    def __getitem__(self, idx):
        if self.use_sampling:
            idx = self.sample_idx()
        ### Get Paths ###
        current_case = self.dataframe.loc[idx]
        input_path = current_case['Input Path']
        nucleus_path = current_case['Nucleus Path']
        mitochondria_path = current_case['Mitochondria Path']
        tubulin_path = current_case['Tubulin Path']
        actin_path = current_case['Actin Path']
        ### Load Image ###

        image_array = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path / input_path)).astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :, :]
            
        image_tensor = tc.from_numpy(image_array)[0, :, :, :, :]
        image_tensor = normalization(image_tensor)

        ### Parse Metadata ###
        if self.return_metadata:
            metadata = self.parse_metadata()

        ### Load Ground-Truth ###
        nucleus_array, nucleus_available = self.load_ground_truth(image_array, nucleus_path)
        mitochondria_array, mitochondria_available = self.load_ground_truth(image_array, mitochondria_path)
        tubulin_array, tubulin_available = self.load_ground_truth(image_array, tubulin_path)
        actin_array, actin_available = self.load_ground_truth(image_array, actin_path)
        
        nucleus_array = percentile_normalization_no_scale(nucleus_array)
        mitochondria_array = percentile_normalization_no_scale(mitochondria_array)
        tubulin_array = percentile_normalization_no_scale(tubulin_array)
        actin_array = percentile_normalization_no_scale(actin_array)
        
        nucleus_tensor = tc.from_numpy(nucleus_array.astype(np.float32))[0, :, :, :, :]
        mitochondria_tensor = tc.from_numpy(mitochondria_array.astype(np.float32))[0, :, :, :, :]
        tubulin_tensor = tc.from_numpy(tubulin_array.astype(np.float32))[0, :, :, :, :]
        actin_tensor = tc.from_numpy(actin_array.astype(np.float32))[0, :, :, :, :]
        ground_truth_tensor = tc.cat((nucleus_tensor, mitochondria_tensor, tubulin_tensor, actin_tensor), dim=0)
        gt_availability = tc.tensor([nucleus_available, mitochondria_available, tubulin_available, actin_available])
        
        ### Apply Augmentation ###
        if self.transforms is not None:
            pass

        ### Get Patches ###
        output_image = tc.zeros((self.num_patches, image_tensor.shape[0], self.patch_size[0], self.patch_size[1])).to(image_tensor.dtype)
        output_ground_truth = tc.zeros((self.num_patches, ground_truth_tensor.shape[0], self.patch_size[0], self.patch_size[1])).to(ground_truth_tensor.dtype)
        output_gt_availability = gt_availability.unsqueeze(0).repeat(self.num_patches, 1)
        for i in range(self.num_patches):
            e_x = image_tensor.shape[3] - self.patch_size[1]
            e_y = image_tensor.shape[2] - self.patch_size[0]
            if e_x == 0 or e_y == 0:
                output_image[i] = image_tensor[:, :, :, ]
                output_ground_truth[i] = ground_truth_tensor[:, 0, :, :]
            else:
                b_x = random.randrange(0, e_x)
                b_y = random.randrange(0, e_y)
                output_image[i] = image_tensor[:, :, b_y:b_y+self.patch_size[0], b_x:b_x+self.patch_size[1]]
                output_ground_truth[i] = ground_truth_tensor[:, 0, b_y:b_y+self.patch_size[0], b_x:b_x+self.patch_size[1]]
        
        ### Return Data for Batch ###
        if self.return_metadata:
            if self.return_input_path:
                return output_image, output_ground_truth, output_gt_availability, metadata, input_path
            else:
                return output_image, output_ground_truth, output_gt_availability, metadata
        else:
            if self.return_input_path:
                return output_image, output_ground_truth, output_gt_availability, input_path
            else:
                return output_image, output_ground_truth, output_gt_availability


def collate_batches(batches):
    for i in range(len(batches)):
        if i == 0:
            output_images = batches[i][0]
            output_ground_truths = batches[i][1]
            output_gt_availabilities = batches[i][2]
        else:
            output_images = tc.cat((output_images, batches[i][0]), dim=0)
            output_ground_truths = tc.cat((output_ground_truths, batches[i][1]), dim=0)
            output_gt_availabilities = tc.cat((output_gt_availabilities, batches[i][2]), dim=0)
    return output_images, output_ground_truths, output_gt_availabilities