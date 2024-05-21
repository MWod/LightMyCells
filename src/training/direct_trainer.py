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
import matplotlib.pyplot as plt
import torchio as tio
import lightning as pl
import PIL
import torchvision.transforms as transforms
import io

### Internal Imports ###
from helpers import objective_functions as of

########################



class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### Models ###
        self.encoder : tc.nn.Module = training_params['encoder']
        self.decoder : tc.nn.Module = training_params['decoder'] 
        ### General Params ###
        self.learning_rate : float = training_params['learning_rate']
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.lr_decay : float = training_params['lr_decay']
        self.log_image_iters : Iterable[int] = training_params['log_image_iters']
        self.number_of_images_to_log : int = training_params['number_of_images_to_log']
        self.patch_size : tuple = training_params['patch_size']
        self.patch_overlap : tuple = training_params['patch_overlap']
        self.echo : bool = training_params['echo']
        ### Objective Functions ###
        self.objective_function : Callable = training_params['objective_function']
    
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        model_list = tc.nn.ModuleList([self.encoder, self.decoder])
        optimizer = tc.optim.AdamW(model_list.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: self.lr_decay ** epoch),
            "frequency": 1,
            "interval": "epoch",
        }
        dict = {'optimizer': optimizer, "lr_scheduler": scheduler}
        return dict
    
    def training_step(self, batch, batch_idx):
        ### Get Batch ###
        image = batch['image']['data'] # Bx1x1xYxX
        ground_truth = batch['ground_truth']['data'] # Bx4x1xYxX
        image = image[:, :, 0, :, :] # Bx1xYxX
        ground_truth = ground_truth[:, :, 0, :, :] # Bx4xYxX

        embedding = self.encoder(image)
        objective = self.decoder(embedding)
        
        if tc.any(tc.isnan(objective)):
            print(f"NaN in the output.")
            return None
        
        objective_loss = self.objective_function(objective, ground_truth)
        loss = objective_loss

        mae = of.mean_absolute_error(objective, ground_truth).item()
        mse = of.mean_squared_error(objective, ground_truth).item()
        ssim = of.structural_similarity_index_measure(objective, ground_truth).item()
        pcc = of.pearson_correlation_coefficient(objective, ground_truth).item()
        cd = of.cosine_distance(objective, ground_truth).item()
                
        ### Log Losses ###
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/objective_loss", objective_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)     
        self.log("Loss/Training/mae", mae, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/mse", mse, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/ssim", ssim, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/pcc", pcc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/cd", cd, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True) 
        return loss
                        
    def validation_step(self, batch, batch_idx):
        ### Process Cases ###
        with tc.no_grad():
            ### Get Batch ###
            image = batch['image']['data']# Bx1x1xYxX
            ground_truth = batch['ground_truth']['data'] # Bx4x1xYxX        
            print(f"Image shape: {image.shape}") if self.echo else None
            ### Create Losses & Counters ###
            outer_objective_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mae = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mse = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_ssim = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_pcc = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_cd = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            ### Iterate Over Batch ###
            for idx in range(image.shape[0]):
                current_input = image[idx:idx+1, :, 0, :, :]
                print(f"Current input shape: {current_input.shape}") if self.echo else None
                current_subject = tio.Subject(input = tio.ScalarImage(tensor=current_input))
                current_ground_truth = ground_truth[idx:idx+1, :, 0, :, :]
                print(f"Current ground-truth shape: {current_ground_truth.shape}") if self.echo else None
                grid_sampler = tio.inference.GridSampler(
                    current_subject,
                    self.patch_size,
                    self.patch_overlap,
                )
                patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=32)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")
                for patches_batch in patch_loader:
                    input_tensor = patches_batch['input'][tio.DATA].to(ground_truth.device)
                    locations = patches_batch[tio.LOCATION]
                    embedding = self.encoder(input_tensor[:, 0, :, :, :])
                    objective = self.decoder(embedding)
                    outputs = objective
                    outputs = outputs.unsqueeze(0).permute(1, 2, 0, 3, 4)
                    aggregator.add_batch(outputs, locations)
                    
                output_tensor = aggregator.get_output_tensor().to(tc.float32).permute(1, 0, 2, 3).to(ground_truth.device)
                print(f"Ground-truth shape: {current_ground_truth.shape}") if self.echo else None
                print(f"Ground-truth dtype: {current_ground_truth.dtype}") if self.echo else None
                print(f"Output shape: {output_tensor.shape}") if self.echo else None
                print(f"Output dtype: {output_tensor.dtype}") if self.echo else None
                
                objective_loss = self.objective_function(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                outer_objective_loss += objective_loss 
                
                outer_mae += of.mean_absolute_error(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                outer_mse += of.mean_squared_error(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                outer_ssim += of.structural_similarity_index_measure(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                outer_pcc += of.pearson_correlation_coefficient(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                outer_cd += of.cosine_distance(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    
                if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
                    self.log_images(current_input, output_tensor, current_ground_truth, batch_idx)
                    
        ### Log Losses ###        
        outer_objective_loss = outer_objective_loss / image.shape[0]
        outer_mae = outer_mae / image.shape[0]     
        outer_mse = outer_mse / image.shape[0]     
        outer_ssim = outer_ssim / image.shape[0]     
        outer_pcc = outer_pcc / image.shape[0]     
        outer_cd = outer_cd / image.shape[0]         
        loss = outer_objective_loss
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/objective_loss", outer_objective_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mae", outer_mae, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mse", outer_mse, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim", outer_ssim, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc", outer_pcc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd", outer_cd, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

    
    def log_images(self, image, output, ground_truth, i) -> None:
        buf = show_images(image, output, ground_truth)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image).unsqueeze(0)[0]
        title = f"Case: {i}, Iter: {str(self.current_epoch)}"
        self.logger.experiment.add_image(title, image, 0)
        plt.close('all')

    
    
def show_images(image, output, ground_truth, return_buffer=True, show=False, suptitle=None):

    plt.figure(dpi=300)
    
    plt.subplot(1, 3, 1)
    plt.imshow(image[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Input")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("GT")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Pre")
    plt.axis('off')
    
    plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle)
    
    if show:
        plt.show()

    if return_buffer:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf

    
class LightningDataModule(pl.LightningDataModule):
    def __init__(self, training_dataloader, validation_dataloader):
        super().__init__()
        self.td = training_dataloader
        self.vd = validation_dataloader
        
    def train_dataloader(self):
        return self.td
    
    def val_dataloader(self):
        return self.vd
    
class LightMyCellsTrainer():
    def __init__(self, **training_params : dict):
        ### General params
        self.training_dataloader : tc.utils.data.DataLoader = training_params['training_dataloader']
        self.validation_dataloader : tc.utils.data.DataLoader = training_params['validation_dataloader']
        self.training_params = training_params
        self.lightning_params = training_params['lightning_params']    
        
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        if self.to_load_checkpoint_path is None:
            self.module = LightningModule(self.training_params, self.lightning_params)
        else:
            self.load_checkpoint()
            
        self.trainer = pl.Trainer(**self.lightning_params)
        self.data_module = LightningDataModule(self.training_dataloader, self.validation_dataloader)

    def save_checkpoint(self) -> None:
        self.trainer.save_checkpoint(pathlib.Path(self.checkpoints_path) / "Last_Iteration")

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path, training_params=self.training_params, lightning_params=self.lightning_params) 

    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()