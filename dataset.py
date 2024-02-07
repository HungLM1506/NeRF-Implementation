import numpy as np
import torch
import os
import torch.nn as nn


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


class Ray(nn.Module):
    pass
