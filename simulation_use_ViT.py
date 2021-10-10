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

"""

if __name__ == "__main__":

    model_params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                       dim=64, depth=1, heads=2, mlp_dim=128)
    model = ViTRegression(**model_params)
    model.load_state_dict(torch.load('tests/test_model_60epochs.pth'))
    model.eval()


    # define environment
    environment = Environment(dt=DETLA_T)

    environment.spawn_agent()

    environment.run(keyboard_steering=False, ai_steering=True, **dict(ai_model=model))
    # test_show_obstacles()
