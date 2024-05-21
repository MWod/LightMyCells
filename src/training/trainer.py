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
        self.reconstruction_decoder : tc.nn.Module = training_params['reconstruction_decoder'] 
        self.nucleus_decoder : tc.nn.Module = training_params['nucleus_decoder'] 
        self.mitochondria_decoder : tc.nn.Module = training_params['mitochondria_decoder'] 
        self.tubulin_decoder : tc.nn.Module = training_params['tubulin_decoder'] 
        self.actin_decoder : tc.nn.Module = training_params['actin_decoder'] 
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
        self.reconstruction_objective_function : Callable = training_params['reconstruction_objective_function']
        self.nucleus_objective_function : Callable = training_params['nucleus_objective_function']
        self.mitochondria_objective_function : Callable = training_params['mitochondria_objective_function']
        self.tubulin_objective_function : Callable = training_params['tubulin_objective_function']
        self.actin_objective_function : Callable = training_params['actin_objective_function']
        self.reconstruction_weight : float = training_params['reconstruction_weight']
        self.nucleus_weight : float = training_params['nucleus_weight']
        self.mitochondria_weight : float = training_params['mitochondria_weight']
        self.tubulin_weight : float = training_params['tubulin_weight']
        self.actin_weight : float = training_params['actin_weight']
    
    def forward(self, x):
        embedding = self.encoder(x)
        nucleus_image = self.nucleus_decoder(embedding)
        mitochondria_image = self.mitochondria_decoder(embedding)
        tubulin_image = self.tubulin_decoder(embedding)
        actin_image = self.actin_decoder(embedding)
        return nucleus_image, mitochondria_image, tubulin_image, actin_image
    
    def configure_optimizers(self):
        model_list = tc.nn.ModuleList([self.encoder,
                                       self.reconstruction_decoder,
                                       self.nucleus_decoder,
                                       self.mitochondria_decoder,
                                       self.tubulin_decoder,
                                       self.actin_decoder])
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
        image = batch['image']['data']# Bx1x1xYxX
        ground_truth = batch['ground_truth']['data'] # Bx4x1xYxX
        gt_availability = batch['gt_availability'] # Bx4 
        image = image[:, :, 0, :, :] # Bx1xYxX
        ground_truth = ground_truth[:, :, 0, :, :] # Bx4xYxX
        ### Calculate Embedding ###
        embedding = self.encoder(image)
        ### Decode If Available ###
        loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=True)
        outer_reconstruction_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
        outer_nucleus_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
        outer_mitochondria_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
        outer_tubulin_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
        outer_actin_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
        num_nucleus = 1.0
        num_mitochondria = 1.0
        num_tubulin = 1.0
        num_actin = 1.0
        for idx in range(image.shape[0]):
            reconstruction = self.reconstruction_decoder(embedding)
            if tc.any(tc.isnan(reconstruction)):
                print(f"NaN in the reconstruction output.")
                return None
            reconstuction_loss = self.reconstruction_objective_function(reconstruction, image)
            outer_reconstruction_loss += reconstuction_loss
            divider = 1.0
            
            if gt_availability[idx, 0]: # Nucleus
                nucleus = self.nucleus_decoder(embedding)
                if tc.any(tc.isnan(nucleus)):
                    print(f"NaN in the nucleus output.")
                    return None
                else:
                    nucleus_loss = self.nucleus_objective_function(nucleus, ground_truth[:, 0:1, :, :])
                    outer_nucleus_loss += nucleus_loss
                    divider += 1.0
                    num_nucleus += 1.0
            else:
                nucleus_loss = 0.0
            if gt_availability[idx, 1]: # Mitochondria
                mitochondria = self.mitochondria_decoder(embedding)
                if tc.any(tc.isnan(mitochondria)):
                    print(f"NaN in the mitochondria output.")
                    return None
                else:
                    mitochondria_loss = self.mitochondria_objective_function(mitochondria, ground_truth[:, 1:2, :, :])
                    outer_mitochondria_loss += mitochondria_loss
                    divider += 1.0
                    num_mitochondria += 1.0
            else:
                mitochondria_loss = 0.0
            if gt_availability[idx, 2]: # Tubulin
                tubulin = self.tubulin_decoder(embedding)
                if tc.any(tc.isnan(tubulin)):
                    print(f"NaN in the tubulin output.")
                    return None
                else:
                    tubulin_loss = self.tubulin_objective_function(tubulin, ground_truth[:, 2:3, :, :])
                    outer_tubulin_loss += tubulin_loss
                    divider += 1.0
                    num_tubulin += 1.0
            else:
                tubulin_loss = 0.0
            if gt_availability[idx, 3]: # Actin
                actin = self.actin_decoder(embedding)
                if tc.any(tc.isnan(actin)):
                    print(f"NaN in the actin output.")
                    return None
                else:
                    actin_loss = self.actin_objective_function(actin, ground_truth[:, 3:4, :, :])
                    outer_actin_loss += actin_loss
                    divider += 1.0
                    num_actin += 1.0
            else:
                actin_loss = 0.0
            loss = loss + (reconstuction_loss + nucleus_loss + mitochondria_loss + tubulin_loss + actin_loss) / divider
        loss = loss / float(image.shape[0])
        outer_reconstruction_loss = outer_reconstruction_loss / float(image.shape[0])
        outer_nucleus_loss = outer_nucleus_loss / num_nucleus
        outer_mitochondria_loss = outer_mitochondria_loss / num_mitochondria
        outer_tubulin_loss = outer_tubulin_loss / num_tubulin
        outer_actin_loss = outer_actin_loss / num_actin
        ### Log Losses ###
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/reconstruction_loss", outer_reconstruction_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/nucleus_loss", outer_nucleus_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/mitochondria_loss", outer_mitochondria_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/tubulin_loss", outer_tubulin_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/actin_loss", outer_actin_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)        
        return loss
                        
    def validation_step(self, batch, batch_idx):
        ### Process Cases ###
        with tc.no_grad():
            ### Get Batch ###
            image = batch['image']['data']# Bx1x1xYxX
            ground_truth = batch['ground_truth']['data'] # Bx4x1xYxX
            gt_availability = batch['gt_availability'] # Bx4            
            print(f"Image shape: {image.shape}") if self.echo else None
            ### Create Losses & Counters ###
            outer_reconstruction_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_nucleus_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mitochondria_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_tubulin_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_actin_loss = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)

            outer_mae_nucleus = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mse_nucleus = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_ssim_nucleus = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_pcc_nucleus = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_cd_nucleus = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)

            outer_mae_mitochondria = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mse_mitochondria = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_ssim_mitochondria = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_pcc_mitochondria = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_cd_mitochondria = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)

            outer_mae_tubulin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mse_tubulin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_ssim_tubulin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_pcc_tubulin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_cd_tubulin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)

            outer_mae_actin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_mse_actin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_ssim_actin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_pcc_actin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)
            outer_cd_actin = tc.tensor([0.0], device=ground_truth.device, requires_grad=False)

            num_nucleus = 1.0
            num_mitochondria = 1.0
            num_tubulin = 1.0
            num_actin = 1.0
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
                    reconstruction = self.reconstruction_decoder(embedding)
                    nucleus = self.nucleus_decoder(embedding)
                    mitochondria = self.mitochondria_decoder(embedding)
                    tubulin = self.tubulin_decoder(embedding)
                    actin = self.actin_decoder(embedding)
                    outputs = tc.cat((nucleus, mitochondria, tubulin, actin, reconstruction), dim=1)
                    outputs = outputs.unsqueeze(0).permute(1, 2, 0, 3, 4)
                    aggregator.add_batch(outputs, locations)
                    
                output_tensor = aggregator.get_output_tensor().to(tc.float32).permute(1, 0, 2, 3).to(ground_truth.device)
                print(f"Ground-truth shape: {current_ground_truth.shape}") if self.echo else None
                print(f"Ground-truth dtype: {current_ground_truth.dtype}") if self.echo else None
                print(f"Output shape: {output_tensor.shape}") if self.echo else None
                print(f"Output dtype: {output_tensor.dtype}") if self.echo else None
                
                # Reconstruction
                reconstuction_loss = self.reconstruction_objective_function(output_tensor[:, 4:5, :, :], image[:, :, 0, :, :])
                outer_reconstruction_loss += reconstuction_loss     
                if gt_availability[idx, 0]: # Nucleus
                    outer_nucleus_loss += self.nucleus_objective_function(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    outer_mae_nucleus += of.mean_absolute_error(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    outer_mse_nucleus += of.mean_squared_error(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    outer_ssim_nucleus += of.structural_similarity_index_measure(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    outer_pcc_nucleus += of.pearson_correlation_coefficient(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    outer_cd_nucleus += of.cosine_distance(output_tensor[:, 0:1, :, :], current_ground_truth[:, 0:1, :, :]).item()
                    num_nucleus = num_nucleus + 1
                if gt_availability[idx, 1]: # Mitochondria
                    outer_mitochondria_loss += self.mitochondria_objective_function(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    outer_mae_mitochondria += of.mean_absolute_error(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    outer_mse_mitochondria += of.mean_squared_error(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    outer_ssim_mitochondria += of.structural_similarity_index_measure(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    outer_pcc_mitochondria += of.pearson_correlation_coefficient(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    outer_cd_mitochondria += of.cosine_distance(output_tensor[:, 1:2, :, :], current_ground_truth[:, 1:2, :, :]).item()
                    num_mitochondria = num_mitochondria + 1
                if gt_availability[idx, 2]: # Tubulin
                    outer_tubulin_loss += self.tubulin_objective_function(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    outer_mae_tubulin += of.mean_absolute_error(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    outer_mse_tubulin += of.mean_squared_error(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    outer_ssim_tubulin += of.structural_similarity_index_measure(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    outer_pcc_tubulin += of.pearson_correlation_coefficient(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    outer_cd_tubulin += of.cosine_distance(output_tensor[:, 2:3, :, :], current_ground_truth[:, 2:3, :, :]).item()
                    num_tubulin = num_tubulin + 1
                if gt_availability[idx, 3]: # Actin
                    outer_actin_loss += self.actin_objective_function(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    outer_mae_actin += of.mean_absolute_error(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    outer_mse_actin += of.mean_squared_error(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    outer_ssim_actin += of.structural_similarity_index_measure(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    outer_pcc_actin += of.pearson_correlation_coefficient(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    outer_cd_actin += of.cosine_distance(output_tensor[:, 3:4, :, :], current_ground_truth[:, 3:4, :, :]).item()
                    num_actin = num_actin + 1
                    
                if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
                    self.log_images(current_input, output_tensor, current_ground_truth, batch_idx)
        ### Log Losses ###   
        outer_reconstruction_loss = outer_reconstruction_loss / image.shape[0]       

        outer_nucleus_loss = outer_nucleus_loss / num_nucleus   
        outer_mae_nucleus = outer_mae_nucleus / num_nucleus
        outer_mse_nucleus = outer_mse_nucleus / num_nucleus
        outer_ssim_nucleus = outer_ssim_nucleus / num_nucleus
        outer_pcc_nucleus = outer_pcc_nucleus / num_nucleus
        outer_cd_nucleus = outer_cd_nucleus / num_nucleus

        outer_mitochondria_loss = outer_mitochondria_loss / num_mitochondria
        outer_mae_mitochondria = outer_mae_mitochondria / num_mitochondria
        outer_mse_mitochondria = outer_mse_mitochondria / num_mitochondria
        outer_ssim_mitochondria = outer_ssim_mitochondria / num_mitochondria
        outer_pcc_mitochondria = outer_pcc_mitochondria / num_mitochondria
        outer_cd_mitochondria = outer_cd_mitochondria / num_mitochondria

        outer_tubulin_loss = outer_tubulin_loss / num_tubulin
        outer_mae_tubulin = outer_mae_tubulin / num_tubulin
        outer_mse_tubulin = outer_mse_tubulin / num_tubulin
        outer_ssim_tubulin = outer_ssim_tubulin / num_tubulin
        outer_pcc_tubulin = outer_pcc_tubulin / num_tubulin
        outer_cd_tubulin = outer_cd_tubulin / num_tubulin

        outer_actin_loss = outer_actin_loss / num_actin 
        outer_mae_actin = outer_mae_actin / num_actin
        outer_mse_actin = outer_mse_actin / num_actin
        outer_ssim_actin = outer_ssim_actin / num_actin
        outer_pcc_actin = outer_pcc_actin / num_actin
        outer_cd_actin = outer_cd_actin / num_actin

        loss = (outer_nucleus_loss + outer_mitochondria_loss + outer_tubulin_loss + outer_actin_loss) / 4.0

        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/objective_loss_nucleus", outer_nucleus_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mae_nucleus", outer_mae_nucleus, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mse_nucleus", outer_mse_nucleus, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_nucleus", outer_ssim_nucleus, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_nucleus", outer_pcc_nucleus, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_nucleus", outer_cd_nucleus, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/objective_loss_tubulin", outer_tubulin_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mae_tubulin", outer_mae_tubulin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mse_tubulin", outer_mse_tubulin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_tubulin", outer_ssim_tubulin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_tubulin", outer_pcc_tubulin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_tubulin", outer_cd_tubulin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/objective_loss_actin", outer_actin_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mae_actin", outer_mae_actin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mse_actin", outer_mse_actin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_actin", outer_ssim_actin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_actin", outer_pcc_actin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_actin", outer_cd_actin, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

        self.log("Loss/Validation/objective_loss_mitochondria", outer_mitochondria_loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mae_mitochondria", outer_mae_mitochondria, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/mse_mitochondria", outer_mse_mitochondria, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/ssim_mitochondria", outer_ssim_mitochondria, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/pcc_mitochondria", outer_pcc_mitochondria, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/cd_mitochondria", outer_cd_mitochondria, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
 

    
    def log_images(self, image, output, ground_truth, i) -> None:
        buf = show_images(image, output, ground_truth)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image).unsqueeze(0)[0]
        title = f"Case: {i}, Iter: {str(self.current_epoch)}"
        self.logger.experiment.add_image(title, image, 0)
        plt.close('all')

    
    
def show_images(image, output, ground_truth, return_buffer=True, show=False, suptitle=None):

    plt.figure(dpi=300)
    
    plt.subplot(2, 5, 1)
    plt.imshow(image[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Input")
    plt.axis('off')
    
    plt.subplot(2, 5, 2)
    plt.imshow(ground_truth[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("GT Nucleus")
    plt.axis('off')
    
    plt.subplot(2, 5, 3)
    plt.imshow(ground_truth[0, 1, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("GT Mitochondria")
    plt.axis('off')
    
    plt.subplot(2, 5, 4)
    plt.imshow(ground_truth[0, 2, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("GT Tubulin")
    plt.axis('off')
    
    plt.subplot(2, 5, 5)
    plt.imshow(ground_truth[0, 3, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("GT Actin")
    plt.axis('off')
    
    
    plt.subplot(2, 5, 6)
    plt.imshow(output[0, 4, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Reconstruction")
    plt.axis('off')
    
    plt.subplot(2, 5, 7)
    plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Pre Nucleus")
    plt.axis('off')
    
    plt.subplot(2, 5, 8)
    plt.imshow(output[0, 1, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Pre Mitochondria")
    plt.axis('off')
    
    plt.subplot(2, 5, 9)
    plt.imshow(output[0, 2, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Pre Tubulin")
    plt.axis('off')
    
    plt.subplot(2, 5, 10)
    plt.imshow(output[0, 3, :, :].detach().cpu().numpy(), cmap='gray')
    plt.title("Pre Actin")
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