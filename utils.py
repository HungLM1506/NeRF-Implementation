import torch
import numpy as np
import os
import math
import imageio
import re


def intrinsic_matrix(fx, fy, ox, oy):
    K = torch.tensor([[fx,  0, ox],
                      [0, fy, oy],
                      [0,  0,  1]])
    return K


def transform(c2w, x_c):
    """
        Camera to World Coordinate Conversion
        c2w: extrinsic matrix
        x_c: position in camera coordinate. Shape [batch_size, height, width, 3]. In final dimension is [x,y,z]
    """
    B, H, W, _ = x_c.shape
    x_c_homogeneous = torch.cat(
        [x_c, torch.ones(B, H, W, 1, device=x_c.device)], dim=-1)  # create [x,y,z] -->[x,y,z,1]

    # batched matmul
    x_w_homogeneous_reshaped = x_c_homogeneous.view(
        B, -1, 4)  # [100, 40000, 4]
    x_w_homogeneous_reshaped = x_w_homogeneous_reshaped.permute(0, 2, 1)
    x_w_homogeneous_reshaped = c2w.bmm(
        x_w_homogeneous_reshaped)  # batch matmul
    x_w_homogeneous = x_w_homogeneous_reshaped.permute(
        0, 2, 1).view(B, H, W, 4)
    x_w = x_w_homogeneous[:, :, :, :3]
    return x_w


def pixel_to_camera(K, uv, s):
    """
        Pixel to Camera Coordinate Conversion
        K: intrinsic matrix 
        uv: position in image coordinate. Shape [batch_size, height,width,C]. [C: image_idx,y,x]
        s: depth of this point along the optical axis. s = Zc
    """
    B, H, W, C = uv.shape
    uv_reshaped = uv.view(B, -1, 3).permute(0, 2, 1)
    uv_homogeneous_reshaped = torch.cat(
        [uv_reshaped[:, 1:], torch.ones((B, 1, H*W), device=uv.device)], dim=1)  # cat [u,v] --> [u,v,1] dim=1 cuz permute previous
    K_inv = torch.inverse(K)
    uv_homogeneous_reshaped = torch.stack(
        (uv_homogeneous_reshaped[:, 1], uv_homogeneous_reshaped[:, 0], uv_homogeneous_reshaped[:, 2]), dim=1)
    x_c_homogeneous_reshaped = K_inv.bmm(uv_homogeneous_reshaped)
    x_c_homogeneous = x_c_homogeneous_reshaped.permute(
        0, 2, 1).view(B, H, W, 3)
    x_c = x_c_homogeneous * s

    return x_c


def pixel_to_rays(K, c2w, uv):
    """
        A rays can be defined by an origin r_0 and a direction r_d. [r_0 is position of camera]
        [To calculate the rays direction for pixel(u,v) we can simply choose a point along this ray with depth equals 1 (s=1)]

        K: intrinsic matrix
        c2w: extrinsic matrix
        uv: position in image coordinate. Shape [batch_size, height,width,C]. [C: image_idx,y,x]
   """
    B, H, W, C = uv.shape  # C = (image_idx,y,x)
    # find x_c with z = 1
    x_c = pixel_to_camera(K, uv, torch.ones((B, H, W, 1), device=uv.device))

    w2c = torch.inverse(c2w)
    R = w2c[:, :3, :3]
    R_inv = torch.inverse(R)
    T = w2c[:, :3, 3]
    # ray origins
    r_o = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)

    # ray directions
    x_w = transform(c2w, x_c)
    r_o = r_o.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
    r_d = (x_w - r_o) / torch.norm((x_w - r_o), dim=-1, keepdim=True)

    return r_o, r_d


def sample_along_rays(r_o, r_d, perturb=True, near=2.0, far=6.0, n_samples=64):
    """
        r_o: origin position of ray
        r_d: direction of ray
        perturb: random deviation of point
        near, far: determine the distance from nearest and farthest
        n_samples: The number of point was create in each ray
    """
    t = torch.linspace(near, far, n_samples, device=r_o.device)
    if perturb:
        t = t + torch.randn_like(t) * (far-near)/n_samples
        # print(f't', t.shape)
    x = r_o + r_d * t.unsqueeze(-1).unsqueeze(-1)  # R = o + td.
    # x is a list
    return x


def positional_encoding(x, L):
    freqs = 2.0 ** torch.arange(L).float().to(x.device)
    x_input = x.unsqueeze(-1) * freqs * 2 * math.pi
    encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
    # add to original input
    encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1)
    return encoding


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def volume_rendering(sigmas, rgbs, step_size):
    # """
    #     Volume rendering function
    #     sigmas: shape [batch_size,n_point,1]
    #     rgbs: rgb is output of model
    #     step_size: distance per jump
    # """
    # B, N, _ = sigmas.shape

    # # transmittance of first ray is 1
    # T_i = torch.cat([torch.ones((B, 1, 1), device=sigmas.device),
    #                 torch.exp(-step_size*torch.cumsum(sigmas, dim=1)[:, :-1])], dim=1)  # output is a list
    # alpha = 1 - torch.exp(-sigmas*step_size)
    # weights = alpha * T_i

    # rendered_colors = torch.sum(weights*rgbs, dim=1)  # output is a number

    # return rendered_colors
    # received help from ChatGPT here to figure out cumsum
    B, N, _ = sigmas.shape
    # transmittance of first ray is 1
    T_i = torch.cat([torch.ones((B, 1, 1), device=rgbs.device),
                    torch.exp(-step_size * torch.cumsum(sigmas, dim=1)[:, :-1])], dim=1)
    alpha = 1 - torch.exp(-sigmas * step_size)
    weights = alpha * T_i

    # accumulated_transmittance = torch.prod(1 - alpha, dim=1, keepdim=True)

    # + accumulated_transmittance.squeeze(1) * torch.ones((B, 3), device=rgbs.device)
    rendered_colors = torch.sum(weights * rgbs, dim=1)
    return rendered_colors


def create_gif(image_folder, gif_path, duration=5):
    def sort_key(filename):
        number = re.search(r"(\d+)", filename)
        return int(number.group(1)) if number else 0

    filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')],
                       key=sort_key)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(gif_path, images, duration=duration)
