### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import pathlib

### External Imports ###
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk

### Internal Imports ###
from paths import paths as p
from aicsimageio.readers.bioformats_reader import BioFile

########################



def parse_to_mha(input_data_path, output_data_path, input_csv_path, output_csv_path):
    input_dataframe = pd.read_csv(input_csv_path)
    output_dataset = []
    for idx in range(len(input_dataframe)):
        current_case = input_dataframe.loc[idx]
        input_path = current_case['Input Path']
        nucleus_path = current_case['Nucleus Path']
        mitochondria_path = current_case['Mitochondria Path']
        tubulin_path = current_case['Tubulin Path']
        actin_path = current_case['Actin Path']
        
        ### Parse Paths ###
        output_path = input_path.replace('.ome.tiff', '.mha')
        if type(nucleus_path) is str:
            output_nucleus_path = nucleus_path.replace('.ome.tiff', '.mha')
        else:
            output_nucleus_path = "None"
            
        if type(mitochondria_path) is str:
            output_mitochondria_path = mitochondria_path.replace('.ome.tiff', '.mha')
        else:
            output_mitochondria_path = "None"
            
        if type(tubulin_path) is str:
            output_tubulin_path = tubulin_path.replace('.ome.tiff', '.mha')
        else:
            output_tubulin_path = "None"
            
        if type(actin_path) is str:
            output_actin_path =actin_path.replace('.ome.tiff', '.mha')
        else:
            output_actin_path = "None"
           
        print(f"Output path: {output_path}") 
        print(f"Output nucleus path: {output_nucleus_path}")
        print(f"Output mitochondria path: {output_mitochondria_path}")
        print(f"Output tubulin path: {output_tubulin_path}")
        print(f"Output actin path: {output_actin_path}")
            
        ### Parse to MHA ###
        array = BioFile(input_data_path / input_path).to_numpy().astype(np.uint16)[0, 0, 0]
        if not os.path.exists(os.path.dirname(output_data_path / output_path)):
            os.makedirs(os.path.dirname(output_data_path / output_path))
        sitk.WriteImage(sitk.GetImageFromArray(array), str(output_data_path / output_path))
        
        if output_nucleus_path != "None":
            array = BioFile(input_data_path / nucleus_path).to_numpy().astype(np.uint16)[0, 0, 0]
            if not os.path.exists(os.path.dirname(output_data_path / output_path)):
                os.makedirs(os.path.dirname(output_data_path / output_path))
            sitk.WriteImage(sitk.GetImageFromArray(array), str(output_data_path / output_nucleus_path))
        
        if output_mitochondria_path != "None":
            array = BioFile(input_data_path / mitochondria_path).to_numpy().astype(np.uint16)[0, 0, 0]
            if not os.path.exists(os.path.dirname(output_data_path / output_path)):
                os.makedirs(os.path.dirname(output_data_path / output_path))
            sitk.WriteImage(sitk.GetImageFromArray(array), str(output_data_path / output_mitochondria_path))
        
        if output_tubulin_path != "None":
            array = BioFile(input_data_path / tubulin_path).to_numpy().astype(np.uint16)[0, 0, 0]
            if not os.path.exists(os.path.dirname(output_data_path / output_path)):
                os.makedirs(os.path.dirname(output_data_path / output_path))
            sitk.WriteImage(sitk.GetImageFromArray(array), str(output_data_path / output_tubulin_path))
        
        if output_actin_path != "None":
            array = BioFile(input_data_path / actin_path).to_numpy().astype(np.uint16)[0, 0, 0]
            if not os.path.exists(os.path.dirname(output_data_path / output_path)):
                os.makedirs(os.path.dirname(output_data_path / output_path))
            sitk.WriteImage(sitk.GetImageFromArray(array), str(output_data_path / output_actin_path))

        image_id = current_case['Image ID']
        modality = current_case['Modality']
        to_append = (image_id, output_path, output_nucleus_path, output_mitochondria_path, output_tubulin_path, output_actin_path, modality)
        output_dataset.append(to_append)
        
        # if idx >= 10:
        #     break
        
    output_dataframe = pd.DataFrame(output_dataset, columns=['Image ID', 'Input Path', 'Nucleus Path', 'Mitochondria Path', 'Tubulin Path', 'Actin Path', 'Modality'])
    if not os.path.isdir(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))
    output_dataframe.to_csv(output_csv_path, index=False)


def run():
    parse_to_mha(p.raw_data_path / "RAW", p.data_path / "MHA", p.data_path / "training_dataset.csv", p.data_path / "training_dataset_mha.csv")
    parse_to_mha(p.raw_data_path / "RAW", p.data_path / "MHA", p.data_path / "validation_dataset.csv", p.data_path / "validation_dataset_mha.csv")


if __name__ == "__main__":
    run()