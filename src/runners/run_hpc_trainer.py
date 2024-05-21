### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib
import argparse

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt
import torchio as tio
import lightning as pl
import PIL
import torchvision.transforms as transforms


from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

### Internal Imports ###
# from experiments import hpc_experiments as hpc
from training import trainer as t
from paths import hpc_paths as p
########################



def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['lightning_params']['max_epochs']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, every_n_epochs=save_step, filename='{epoch}', save_top_k=-1)
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor='Loss/Validation/loss')
    best_nucleus_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_nucleus_loss', save_top_k=1, mode='min', monitor='Loss/Validation/nucleus_loss')
    best_mitochondria_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_mitochondria_loss', save_top_k=1, mode='min', monitor='Loss/Validation/mitochondria_loss')
    best_tubulin_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_tubulin_loss', save_top_k=1, mode='min', monitor='Loss/Validation/tubulin_loss')
    best_actin_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_actin_loss', save_top_k=1, mode='min', monitor='Loss/Validation/actin_loss')
    checkpoints_iters = list(range(0, num_iterations, save_step))
    log_image_iters = list(range(0, num_iterations, save_step))
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    log_dir = os.path.join(p.logs_path, experiment_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [checkpoint_callback, best_loss_checkpoint, best_nucleus_checkpoint, best_mitochondria_checkpoint, best_tubulin_checkpoint, best_actin_checkpoint] 
    training_params['checkpoints_path'] = checkpoints_path
    training_params['checkpoint_iters'] = checkpoints_iters
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params):
    training_params = initialize(training_params)
    trainer = t.LightMyCellsTrainer(**training_params)
    trainer.run()
    
def run():
    pass

def run_named_experiment(function_name):
    config = getattr(hpc, function_name)()
    run_training(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()
    function_name = args.experiment
    print(f"Function name: {function_name}")
    if function_name is None:
        run()
    else:
        run_named_experiment(function_name)