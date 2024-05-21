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

### Internal Imports ###
from paths import paths as p

########################


def get_unique_ids(images):
    ids = []
    for image in images:
        id = image.split("_")[1]
        ids.append(int(id))
    ids = list(set(ids))
    return ids

def get_magnifications(images, id):
    names = []
    for image in images:
        if int(image.split("_")[1]) == id and "z" in image:
            names.append(image)
    return names
            
def get_modality(image):
    modality = image.split("_")[2]
    return modality

    
def create_csv_1(data_path, output_csv_path, study_ids):
    dataset = []

    for study_id in study_ids:
        case_path = data_path / f"Study_{str(study_id)}"
        images = os.listdir(case_path)
        unique_ids = get_unique_ids(images)
        for unique_id in unique_ids:
            current_images = get_magnifications(images, unique_id)
            for current_image in current_images:
                ### Build Input Path ###
                input_path = pathlib.Path(f"Study_{str(study_id)}") / current_image
                modality = get_modality(current_image)
                ### Create Ground-Truth Paths ###
                nucleus_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Nucleus.ome.tiff"
                if not os.path.exists(data_path / nucleus_path):
                    nucleus_path = "None"
                mitochondria_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Mitochondria.ome.tiff"
                if not os.path.exists(data_path / mitochondria_path):
                    mitochondria_path = "None"
                tubulin_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Tubulin.ome.tiff"
                if not os.path.exists(data_path / tubulin_path):
                    tubulin_path = "None"
                actin_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Actin.ome.tiff"
                if not os.path.exists(data_path / actin_path):
                    actin_path = "None"
                to_append = (input_path, nucleus_path, mitochondria_path, tubulin_path, actin_path, modality)
                dataset.append(to_append)

    dataframe = pd.DataFrame(dataset, columns=['Input Path', 'Nucleus Path', 'Mitochondria Path', 'Tubulin Path', 'Actin Path', 'Modality'])
    if not os.path.isdir(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))
    dataframe.to_csv(output_csv_path, index=False)    


def create_csv_2(data_path, output_csv_path, output_ids_path, study_ids):
    image_ids = []
    dataset = []

    for study_id in study_ids:
        case_path = data_path / f"Study_{str(study_id)}"
        images = os.listdir(case_path)
        unique_ids = get_unique_ids(images)
        for unique_id in unique_ids:
            current_images = get_magnifications(images, unique_id)
            for current_image in current_images:
                ### Build Input Path ###
                input_path = pathlib.Path(f"Study_{str(study_id)}") / current_image
                modality = get_modality(current_image)
                ### Create Ground-Truth Paths ###
                nucleus_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Nucleus.ome.tiff"
                if not os.path.exists(data_path / nucleus_path):
                    nucleus_path = "None"
                mitochondria_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Mitochondria.ome.tiff"
                if not os.path.exists(data_path / mitochondria_path):
                    mitochondria_path = "None"
                tubulin_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Tubulin.ome.tiff"
                if not os.path.exists(data_path / tubulin_path):
                    tubulin_path = "None"
                actin_path = pathlib.Path(f"Study_{str(study_id)}") / f"image_{unique_id}_Actin.ome.tiff"
                if not os.path.exists(data_path / actin_path):
                    actin_path = "None"
                to_append = (unique_id, input_path, nucleus_path, mitochondria_path, tubulin_path, actin_path, modality)
                dataset.append(to_append)
                image_ids.append(unique_id)

    dataframe = pd.DataFrame(dataset, columns=['Image ID', 'Input Path', 'Nucleus Path', 'Mitochondria Path', 'Tubulin Path', 'Actin Path', 'Modality'])
    if not os.path.isdir(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))
    dataframe.to_csv(output_csv_path, index=False)
    unique_ids = np.array(list(set(image_ids)))
    np.save(str(output_ids_path), unique_ids)


def split_ids(input_ids_path, output_training_ids, output_validation_ids, split_ratio):
    ids = np.load(input_ids_path)
    np.random.shuffle(ids)

    number_of_cases = len(ids)
    training_ids = ids[0:int(split_ratio*number_of_cases)]
    validation_ids = ids[int(split_ratio*number_of_cases):]
    print(f"Number of IDs: {number_of_cases}")
    print(f"Number of training IDs: {len(training_ids)}")
    print(f"Number of validation IDs: {len(validation_ids)}")

    np.save(str(output_training_ids), training_ids)
    np.save(str(output_validation_ids), validation_ids)

def split_dataframe(input_csv_path, output_csv_training_path, output_csv_validation_path, train_ids, val_ids):
    dataframe = pd.read_csv(input_csv_path)
    train_ids = np.load(train_ids)
    val_ids = np.load(val_ids)

    training_dataframe = dataframe.copy()
    validation_dataframe = dataframe.copy()

    print(f"Dataframe Length: {len(dataframe)}")
    print(f"Number of training IDs: {len(train_ids)}")
    print(f"Number of validation IDs: {len(val_ids)}")

    for id in val_ids:
        training_dataframe = training_dataframe.drop(training_dataframe[training_dataframe['Image ID'] == id].index)

    for id in train_ids:
        validation_dataframe = validation_dataframe.drop(validation_dataframe[validation_dataframe['Image ID'] == id].index)
        
    print(f"Training dataframe length: {len(training_dataframe)}")
    print(f"Validation dataframe length: {len(validation_dataframe)}")

    training_dataframe.to_csv(output_csv_training_path, index=False)
    validation_dataframe.to_csv(output_csv_validation_path, index=False)



def split_dataframe_2(input_csv_path, output_csv_path, num_cases):
    dataframe = pd.read_csv(input_csv_path)
    dataframe = dataframe.sample(frac=1, random_state=1234)
    print(f"Dataframe Length: {len(dataframe)}")
    output_dataframe = dataframe[:num_cases]
    output_dataframe.to_csv(output_csv_path)

    
    
### Old Version - Produces Strongly Imbalanced Representation (Study-Level) ###
def create_training_csv_1():
    data_path = p.raw_data_path
    output_csv_path = p.data_path / "training_dataset.csv"
    study_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29]
    create_csv_1(data_path, output_csv_path, study_ids)

def create_validation_csv_1():
    data_path = p.raw_data_path
    output_csv_path = p.data_path / "validation_dataset.csv"
    study_ids = [3, 7, 13, 21, 28, 30]
    create_csv_1(data_path, output_csv_path, study_ids)


### New Version - Better Balance (Image-Level) ###
def create_dataset_1():
    data_path = p.raw_data_path
    output_csv_path = p.data_path / "dataset.csv"
    output_ids_path = p.data_path / "ids.npy"
    study_ids = list(range(1, 31))
    create_csv_2(data_path, output_csv_path, output_ids_path, study_ids)

def run():
    pass # create 5-fold splits

if __name__ == "__main__":
    run()