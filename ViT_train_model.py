import math
import os
import random
import sys
import time

import numpy as np
import pybullet as pb
import pybullet_data
import torch
from PIL import Image
from matplotlib import pyplot as plt

from Environment import Environment
from ViTModelGenerator import ViTRegression

DETLA_T = 0.1

#TODO:
"""
[ ] - 
"""

if __name__ == "__main__":
    training_data_dir_path = 'model_training/data/10-10_21:30_training_data/'

    params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                       dim=64, depth=1, heads=2, mlp_dim=128)

    model = ViTRegression(**params)
    model.load_dataloaders(dir_path=training_data_dir_path)
    model.train_epochs(max_epochs=100, save_path='10_10_model.pth')
    model.plot_training_history()
