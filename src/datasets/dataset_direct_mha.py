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











class DirectDataset(tio.SubjectsDataset):
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
        gt_mode : str = "nucleus", # nucleus, mitochondria, tubulin, actin
        intensity_transforms : Callable = None,
        normalization_mode='min_max',#min_max, percentile
        gt_normalization_mode='percentile',
        use_sampling = False,
        ids_to_no_augment = None): 
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.gt_mode = gt_mode
        self.iteration_size = iteration_size
        self.transforms = transforms
        self.intensity_transforms = intensity_transforms
        self.return_metadata = return_metadata
        self.return_input_path = return_input_path
        self.normalization_mode = normalization_mode
        self.use_sampling = use_sampling
        self.gt_normalization_mode = gt_normalization_mode
        self.ids_to_no_augment = ids_to_no_augment
        self.dataframe = pd.read_csv(self.csv_path)
        if self.gt_mode == "nucleus":
            self.dataframe = self.dataframe[self.dataframe['Nucleus Path'].notna()].reset_index(drop=True)
        elif self.gt_mode == "mitochondria":
            self.dataframe = self.dataframe[self.dataframe['Mitochondria Path'].notna()].reset_index(drop=True)
        elif self.gt_mode == "tubulin":
            self.dataframe = self.dataframe[self.dataframe['Tubulin Path'].notna()].reset_index(drop=True)
        elif self.gt_mode == "actin":
            self.dataframe = self.dataframe[self.dataframe['Actin Path'].notna()].reset_index(drop=True)
        else:
            raise ValueError("Unsupported GT mode.")
        
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)
            
        self.ids = np.arange(len(self.dataframe))
        
        if self.iteration_size > len(self.dataframe):
            self._subjects = [lambda: self.__getitem__(idx) for idx in range(len(self.dataframe))]
        elif self.iteration_size > 0 and self.iteration_size < len(self.dataframe):
            self._subjects = [lambda: self.__getitem__(idx) for idx in range(self.iteration_size)]
        else:
            self._subjects = [lambda: self.__getitem__(idx) for idx in range(len(self.dataframe))]
            
        self.weights = self.get_weights()

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def get_weights(self):
        weights = []
        study_dict = {}
        study_numbers = {}
        for idx in range(len(self.dataframe)):
            current_case = self.dataframe.iloc[idx]
            input_path = current_case['Input Path']
            study = int(str(input_path).split("Study_")[1].split("/")[0])
            study_dict[idx] = study
            try:
                study_numbers[study] += 1
            except:
                study_numbers[study] = 1
        print(f"Study numbers: {study_numbers}")
        for idx in range(len(self.dataframe)):
            weights.append(1 / study_numbers[study_dict[idx]])
            
        weights = np.array(weights)
        norm = np.linalg.norm(weights, ord=1)
        weights = weights / norm
        return weights
            
    
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def parse_metadata(self, image):
        metadata = {}
        # TODO
        return metadata

    def load_ground_truth(self, array, relative_path):
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path / relative_path)).astype(np.uint16)[np.newaxis, np.newaxis, np.newaxis, :, :]
        return ground_truth
    
    def sample_idx(self):
        idx = np.random.choice(self.ids, size=1, p=self.weights)[0]
        return idx
        
    def __getitem__(self, idx):
        if self.use_sampling:
            idx = self.sample_idx()
        ### Get Paths ###
        current_case = self.dataframe.loc[idx]
        input_path = current_case['Input Path']
        if self.gt_mode == "nucleus":
            gt_path = current_case['Nucleus Path']
        elif self.gt_mode == "mitochondria":
            gt_path = current_case['Mitochondria Path']
        elif self.gt_mode == "tubulin":
            gt_path = current_case['Tubulin Path']
        elif self.gt_mode == "actin":
            gt_path = current_case['Actin Path']
        else:
            raise ValueError("Unsupported GT mode.")
        ### Load Image ###

        image_array = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path / input_path)).astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :, :]

        if self.normalization_mode == "min_max":
            image_tensor = tc.from_numpy(image_array)[0, :, :, :, :]
            image_tensor = normalization(image_tensor)
        elif self.normalization_mode == "percentile":
            image_array = percentile_normalization_no_scale(image_array)[0]
            image_tensor = tc.from_numpy(image_array.astype(np.float32))[:, :, :, :]
        elif self.normalization_mode == "value":
            image_tensor = tc.from_numpy(image_array)[0, :, :, :, :]
            image_tensor = normalization(image_tensor)
            image_tensor = image_tensor / 10_000
        else:
            image_tensor = tc.from_numpy(image_array)[0, :, :, :, :]
        
        ### Parse Metadata ###
        if self.return_metadata:
            metadata = self.parse_metadata()

        ### Load Ground-Truth ###
        gt_array = self.load_ground_truth(image_array, gt_path)
        if self.gt_normalization_mode == "percentile":
            gt_array = percentile_normalization_no_scale(gt_array)[0]
        elif self.gt_normalization_mode == "min_max":
            gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))
            gt_array = gt_array[0]
        elif self.gt_normalization_mode == "value":
            gt_array = gt_array[0] / 10_000
        else:
            gt_array = gt_array[0]
        ground_truth_tensor = tc.from_numpy(gt_array.astype(np.float32))[:, :, :, :]
        
        ### Create Subject ###
        subject = tio.Subject(
            image = tio.ScalarImage(tensor=image_tensor),
            ground_truth = tio.ScalarImage(tensor=ground_truth_tensor),
        )
        
        ### Apply Augmentation ###
        if self.transforms is not None:
            if self.ids_to_no_augment is None:
                subject = self.transforms(subject)
            else:
                augment = True
                for id_to_exclude in self.ids_to_no_augment:
                    if f"Study_{int(id_to_exclude)}" in input_path:
                        augment = False
                        break
                if augment:
                    print(input_path)
                    subject = self.transforms(subject)
                    
            
        if self.intensity_transforms is not None:
            intensity_subject = tio.Subject(
                image = tio.ScalarImage(tensor=subject['image']['data'])
            )
            intensity_subject = self.intensity_transforms(intensity_subject)
            
            if self.normalization_mode == "min_max":
                tensor_normalized = normalization(intensity_subject['image']['data'])
            elif self.normalization_mode == "percentile":
                tensor_normalized = tc.from_numpy(percentile_normalization_no_scale(intensity_subject['image']['data'].numpy()))
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=tensor_normalized),
                ground_truth = tio.ScalarImage(tensor=subject['ground_truth']['data'])
            )
            
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