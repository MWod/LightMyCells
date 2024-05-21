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
from aicsimageio.readers.bioformats_reader import BioFile
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



# class TrainingDataset(tc.utils.data.Dataset):
class TrainingDataset(tio.SubjectsDataset):
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
        load_mode = "tiff"):
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
        self.load_mode = load_mode # "tiff", "mha"
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)
            
        self.ids = np.arange(len(self.dataframe))
        self.nucleus_weight = nucleus_weight
        self.mitochondria_weight = mitochondria_weight
        self.tubulin_weight = tubulin_weight
        self.actin_weight = actin_weight
        self.weights = self.get_weights()
        
        if self.iteration_size > len(self.dataframe):
            self._subjects = [lambda idx: self.__getitem__(idx) for idx in range(len(self.dataframe))]
        elif self.iteration_size > 0 and self.iteration_size < len(self.dataframe):
            self._subjects = [lambda idx: self.__getitem__(idx) for idx in range(self.iteration_size)]
        else:
            self._subjects = [lambda idx: self.__getitem__(idx) for idx in range(len(self.dataframe))]

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
            if self.load_mode == "tiff":
                ground_truth = BioFile(self.data_path / relative_path).to_numpy().astype(np.uint16)
            elif self.load_mode == "mha":
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

        if self.load_mode == "tiff":
            image = BioFile(self.data_path / input_path)
            image_array = image.to_numpy().astype(np.float32)
        elif self.load_mode == "mha":
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

        ### Create Subject ###
        subject = tio.Subject(
            image = tio.ScalarImage(tensor=image_tensor),
            ground_truth = tio.ScalarImage(tensor=ground_truth_tensor),
            gt_availability = gt_availability, 
        )
        
        ### Apply Augmentation ###
        if self.transforms is not None:
            pass
        
        ### Return Data for Batch ###
        if self.return_metadata:
            if self.return_input_path:
                return subject, metadata, input_path
            else:
                return subject, metadata
        else:
            if self.return_input_path:
                return subject, input_path
            else:
                return subject