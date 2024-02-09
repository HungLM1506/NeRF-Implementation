import numpy as np
import torch
import os
import torch.nn as nn
from utils import pixel_to_rays


def load_data_numpy(file_path):
    data = np.load(file_path)

    # training images and normalize
    images_train = data['images_train'] / 255.0  # shape [100,200,200,3]

    # camera for image train
    # camera to world transform matrix
    c2w_train = data['c2ws_train']  # shape [100,4,4]

    # validation images and normalize
    images_valid = data['images_val'] / 255.0  # shape [10,200,200,3]

    # camera for image validation
    # camera to world transform matrix
    c2w_valid = data['c2ws_val']  # shape [10,4,4]

    # test cameras for novel-view video rendering
    # camera to world transform matrix
    c2w_test = data["c2ws_test"]  # shape [60,4,4]

    # camera focal length
    focal = data['focal']  # float

    return images_train, c2w_train, images_valid, c2w_valid, c2w_test, focal


class RaysData:
    def __init__(self, images, K, c2w, device='cuda'):
        self.images = images
        self.K = K
        self.c2w = c2w
        self.device = device

        self.height = images.shape[1]
        self.width = images.shape[2]

        # create UV grid
        self.uv = torch.stack(torch.meshgrid(torch.arange(self.images.shape[0]), torch.arange(
            self.height), torch.arange(self.width)), dim=-1).to(device).float()  # shape(image_idx,u,v)
        # add 0.5 offset to each pixel
        # select to u and v. dimension [0] is image_idx
        self.uv[..., 1] += 0.5
        self.uv[..., 2] += 0.5
        self.uv_flattened = self.uv.reshape(-1, 3)

        self.r_o, self.r_d = pixel_to_rays(K, c2w, self.uv)
        self.pixels = images.reshape(-1, 3)
        self.r_o_flattened = self.r_o.reshape(-1, 3)
        self.r_d_flattened = self.r_d.reshape(-1, 3)

    def sample_rays(self, batch_size):
        # sample rays
        idx = torch.randint(
            0, self.pixels.shape[0], (batch_size,), device=self.pixels.device)
        return self.r_o_flattened[idx], self.r_d_flattened[idx], self.pixels[idx]

    # used for validation
    def sample_rays_single_image(self, image_index=None):
        if image_index is None:
            image_index = torch.randint(
                0, self.c2w.shape[0], (1,), device=self.device).item()
        start_idx = image_index * self.height * self.width
        end_idx = start_idx + self.height * self.width

        r_o_single = self.r_o_flattened[start_idx:end_idx]
        r_d_single = self.r_d_flattened[start_idx:end_idx]
        pixels_single = self.pixels[start_idx:end_idx]

        return r_o_single, r_d_single, pixels_single
