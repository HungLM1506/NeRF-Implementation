import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import sample_along_rays
from utils import volume_rendering
from utils import create_gif


def render_images(model, test_dataset):
    # testing
    model.eval()
    with torch.no_grad():
        for i in range(test_dataset.c2w.shape[0]):
            rays_o, rays_d, _ = test_dataset.sample_rays_single_image(i)
            points = sample_along_rays(rays_o, rays_d)
            points = points.permute(1, 0, 2)
            rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
            rgb, sigmas = model(points, rays_d)
            comp_rgb = volume_rendering(
                sigmas, rgb, step_size=(6.0 - 2.0) / 64)
            # save image
            image = comp_rgb.reshape(200, 200, 3).cpu().numpy()
            plt.imsave(f"final_render/render_{i}.jpg", image)

    create_gif('final_render', 'final_render/training.gif')


# if __name__ ==  "__main__":
#     device = torch.cuda.is_available()
