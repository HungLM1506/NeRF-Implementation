import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import sample_along_rays
from utils import volume_rendering
from utils import psnr
from eval import render_images
from datetime import datetime


def train_model(model, train_dataset, val_dataset, test_dataset, optimizer, criterion, iters=3000, batch_size=10000, device='cuda'):

    psnr_scores = []

    model.train()
    for i in tqdm(range(iters)):
        rays_o, rays_d, pixels = train_dataset.sample_rays(batch_size)
        points = sample_along_rays(rays_o, rays_d, perturb=True)
        points = points.permute(1, 0, 2)
        rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)

        optimizer.zero_grad()
        rgb, sigmas = model(points, rays_d)
        comp_rgb = volume_rendering(sigmas, rgb, step_size=(6.0 - 2.0) / 64)

        loss = criterion(comp_rgb, pixels)
        loss.backward()
        optimizer.step()

        # training PSNR
        print(
            f"Training PSNR: {psnr(comp_rgb.detach().cpu().numpy(), pixels.cpu().numpy())}")

        # validation
        if (i + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                rays_o, rays_d, pixels = val_dataset.sample_rays_single_image()
                points = sample_along_rays(rays_o, rays_d)
                points = points.permute(1, 0, 2)
                rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
                rgb, sigmas = model(points, rays_d)
                comp_rgb = volume_rendering(
                    sigmas, rgb, step_size=(6.0 - 2.0) / 64)
                loss = criterion(comp_rgb, pixels)
                # print(f"Validation loss: {loss}")
                curr_psnr = psnr(comp_rgb.cpu().numpy(), pixels.cpu().numpy())
                print(f"Validation PSNR: {curr_psnr:.2f} dB")
                psnr_scores.append(curr_psnr)
                # save image
                image = comp_rgb.reshape(200, 200, 3).cpu().numpy()
                plt.imsave(f"nerf_output/iter{i+1}.jpg", image)
            model.train()

    # save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(),
               f"checkpoints/nerf_checkpoint_{timestamp}.pt")

    # create PSNR plot
    # plt.figure()
    # plt.plot(range(25, 3001, 25), psnr_scores)
    # plt.xlabel('Iteration')
    # plt.ylabel('PSNR (dB)')
    # plt.title('PSNR vs. Iteration')
    # plt.savefig('plots/psnr_nerf.png')

    render_images(model, test_dataset)
