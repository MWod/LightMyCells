### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import torch as tc
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
import torchio as tio

### Internal Imports ###
from networks import runet, transformer_autoencoder, swinunetr
from networks import configs as cfg
from datasets import dataset_direct_mha
from paths import paths as p
from helpers import objective_functions
from augmentation import transforms as t

########################


def runet_experiment_1():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Direct_Tubulin_1"
    learning_rate = 0.001
    save_step = 5
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return objective_functions.mean_absolute_error(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 51
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params






def runet_experiment_2():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Direct_Tubulin_1"
    learning_rate = 0.001
    save_step = 5
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 51
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params













































def tubulin_experiment_1():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_1"
    learning_rate = 0.001
    save_step = 10
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_1():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_1"
    learning_rate = 0.001
    save_step = 100
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 1001
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_1():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_1"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_1():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = runet.RUNetEncoder(**cfg.default_encoder_config())
    decoder = runet.RUNetDecoder(**cfg.default_decoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_1"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params






























def tubulin_experiment_2():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = transformer_autoencoder.Encoder(**cfg.default_transformer_autoencoder_config())
    decoder = transformer_autoencoder.Decoder(**cfg.default_transformer_autodecoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_2"
    learning_rate = 0.001
    save_step = 10
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_2():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = transformer_autoencoder.Encoder(**cfg.default_transformer_autoencoder_config())
    decoder = transformer_autoencoder.Decoder(**cfg.default_transformer_autodecoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_2"
    learning_rate = 0.001
    save_step = 100
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 1001
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_2():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = transformer_autoencoder.Encoder(**cfg.default_transformer_autoencoder_config())
    decoder = transformer_autoencoder.Decoder(**cfg.default_transformer_autodecoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_2"
    learning_rate = 0.001
    save_step = 5
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_2():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder = transformer_autoencoder.Encoder(**cfg.default_transformer_autoencoder_config())
    decoder = transformer_autoencoder.Decoder(**cfg.default_transformer_autodecoder_config())
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_2"
    learning_rate = 0.001
    save_step = 5
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 0.2*objective_functions.mean_absolute_error(t1, t2) + 0.8*objective_functions.structural_similarity_index_measure(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




























































def tubulin_experiment_3():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_3"
    learning_rate = 0.001
    save_step = 10
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_3():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_3"
    learning_rate = 0.001
    save_step = 100
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 401
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_3():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_3"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_3():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_3"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params

















































































def tubulin_experiment_4():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_1" / 'epoch=71_loss.ckpt'
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
            
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_4"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_4():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_1" / 'epoch=47_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_4"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 301
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_4():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_1" / 'epoch=26_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_4"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_4():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_1" / 'epoch=28_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_4"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params






































def tubulin_experiment_5():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_5"
    learning_rate = 0.001
    save_step = 10
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_5():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_5"
    learning_rate = 0.001
    save_step = 100
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 401
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_5():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_5"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_5():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    encoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    decoder_config = {
        'img_size': (512, 512),
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 2,
        'depths': (2, 2, 4, 4),
        'num_heads': (12, 12, 24, 24),
        'feature_size': 24
    }
    
    encoder = swinunetr.Encoder(**encoder_config)
    decoder = swinunetr.Decoder(**decoder_config)
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_5"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.1 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





































def tubulin_experiment_6():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_1" / 'epoch=71_loss.ckpt'
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
            
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_6"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_6():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_1" / 'epoch=47_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_6"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 301
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_6():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_1" / 'epoch=26_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_6"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_6():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 300
    samples_per_volume = 8
    training_batch_size = 16
    validation_batch_size = 1
    patch_size = (1, 256, 256)
    patch_overlap = (0, 128, 128)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_1" / 'epoch=28_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_6"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params



























































































def tubulin_experiment_7():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 600
    samples_per_volume = 32
    training_batch_size = 64
    validation_batch_size = 1
    patch_size = (1, 128, 128)
    patch_overlap = (0, 64, 64)
    gt_mode = "tubulin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_1" / 'epoch=71_loss.ckpt'
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
            
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_7"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_7():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 600
    samples_per_volume = 32
    training_batch_size = 64
    validation_batch_size = 1
    patch_size = (1, 128, 128)
    patch_overlap = (0, 64, 64)
    gt_mode = "actin"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_1" / 'epoch=47_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_7"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 301
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_7():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 600
    samples_per_volume = 32
    training_batch_size = 64
    validation_batch_size = 1
    patch_size = (1, 128, 128)
    patch_overlap = (0, 64, 64)
    gt_mode = "nucleus"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_1" / 'epoch=26_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_7"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_7():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 600
    samples_per_volume = 32
    training_batch_size = 64
    validation_batch_size = 1
    patch_size = (1, 128, 128)
    patch_overlap = (0, 64, 64)
    gt_mode = "mitochondria"
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_1" / 'epoch=28_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_7"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params












































































def tubulin_experiment_8():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    use_sampling = True
    normalization_mode = "percentile"
    transforms = t.geometric()
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_4" / 'epoch=97_loss.ckpt'
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
            
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_8"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_8():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    use_sampling = True
    normalization_mode = "percentile"
    transforms = t.geometric()
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_4" / 'epoch=14_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_8"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 301
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_8():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    use_sampling = True
    normalization_mode = "percentile"
    transforms = t.geometric()
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_4" / 'epoch=30_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_8"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_8():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    use_sampling = True
    normalization_mode = "percentile"
    transforms = t.geometric()
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_4" / 'epoch=29_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_8"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 31
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params


































































def tubulin_experiment_9():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "tubulin"
    use_sampling = True
    normalization_mode = "min_max"
    gt_normalization_mode = "min_max"
    transforms = None
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Tubulin_4" / 'epoch=97_loss.ckpt'
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
            
    ### Parameters ###
    experiment_name = "LightMyCells_Tubulin_9"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.99
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params





def actin_experiment_9():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "actin"
    use_sampling = True
    normalization_mode = "min_max"
    gt_normalization_mode = "min_max"
    transforms = t.geometric()
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Actin_4" / 'epoch=14_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Actin_9"
    learning_rate = 0.001
    save_step = 50
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.999
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 301
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params







def nucleus_experiment_9():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "nucleus"
    use_sampling = True
    normalization_mode = "min_max"
    gt_normalization_mode = "min_max"
    transforms = None
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Nucleus_4" / 'epoch=30_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Nucleus_9"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 41
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params




def mitochondria_experiment_9():
    """
    """
    ### Datasets ###
    data_path = p.data_path / "MHA"
    training_csv_path = p.data_path / "training_dataset_mha.csv"
    validation_csv_path = p.data_path / "validation_dataset_mha.csv"
    transforms = None
    return_metadata = False
    return_input_path = False
    num_workers = 32
    queue_length = 200
    samples_per_volume = 4
    training_batch_size = 4
    validation_batch_size = 1
    patch_size = (1, 512, 512)
    patch_overlap = (0, 256, 256)
    gt_mode = "mitochondria"
    use_sampling = True
    normalization_mode = "min_max"
    gt_normalization_mode = "min_max"
    transforms = None
    training_dataset = dataset_direct_mha.DirectDataset(data_path, training_csv_path, iteration_size=-1, transforms=transforms, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=use_sampling, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    validation_dataset = dataset_direct_mha.DirectDataset(data_path, validation_csv_path, iteration_size=-1, transforms=None, return_metadata=return_metadata, return_input_path=return_input_path, gt_mode=gt_mode, use_sampling=False, normalization_mode=normalization_mode, gt_normalization_mode=gt_normalization_mode)
    ### Dataloaders ###
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        training_dataset,
        queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
    )
    training_dataloader = tc.utils.data.DataLoader(
        patches_queue,
        batch_size=training_batch_size,
        num_workers=0,
        shuffle=False,
    )
    
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
    print(f"Training set size: {len(training_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    
    ### Models ###
    checkpoint_path = p.checkpoints_path / "LightMyCells_Mitochondria_4" / 'epoch=29_loss.ckpt'
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
    
    ### Parameters ###
    experiment_name = "LightMyCells_Mitochondria_9"
    learning_rate = 0.001
    save_step = 3
    to_load_checkpoint_path = None
    number_of_images_to_log = 10
    lr_decay = 0.97
    optimizer_weight_decay = 0.01
    echo = False
    
    def objective_function(t1, t2):
        return 1.0 * objective_functions.mean_squared_error(t1, t2) + 0.2 * objective_functions.structural_similarity_index_measure(t1, t2) + 0.1 * objective_functions.pearson_correlation_coefficient(t1, t2) + 0.1 * objective_functions.cosine_distance(t1, t2)
    
    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 41
    accumulate_grad_batches = 1
    gradient_clip_val = 1000000000000000000000.0
    reload_dataloaders_every_n_epochs = 100000
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs

    ### Parse Parameters ###
    training_params = dict()
    ### Models ###
    training_params['encoder'] = encoder
    training_params['decoder'] = decoder
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['patch_size'] = patch_size
    training_params['patch_overlap'] = patch_overlap
    training_params['echo'] = echo
    
    
    ### Cost functions and params
    training_params['objective_function'] = objective_function

    ########################################
    return training_params











