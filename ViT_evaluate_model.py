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
from ResnetModelGenerator import ResnetRegression
from ViTModelGenerator import ViTRegression
from ViTModelGenerator import ViTRegression

DETLA_T = 0.1

#TODO:
"""
[ ] - How to use RGB images? For now it is running on 1 channel - Grayscale
[ ] - How to exclude the Agent going off the road?
    - make leaving the lanes same condition as hitting an obstacle
    - rerecord training data
[ ] - How to use the sequential data?
    - reread the paper
[ ] - How to make comparisons? How do we know if this method is better than sth other?
"""

if __name__ == "__main__":

    do_vit = True
    # do_vit = False

    if do_vit:
        params = dict(image_size=256, patch_size=8, num_outputs=1, channels=3,
                      dim=64, depth=1, heads=2, mlp_dim=128)
        config_ViT = dict(model='ViT', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
        model = ViTRegression(wandb_config=config_ViT, **params)
        model.load_state_dict(torch.load('model_training/trained_models/model_20_ViT.pth'))

    else:
        config_resnet = dict(model='resnet', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
        model = ResnetRegression(wandb_config=config_resnet)
        model.load_state_dict(torch.load('model_training/trained_models/model_20_resnet.pth'))

    model.eval()

    # define environment
    ai_steering = True
    difficulty_distance = 30
    environment = Environment(dt=DETLA_T, ai_steering=ai_steering, difficulty_distance=difficulty_distance)
    environment.set_cool_game_vibe_camera_position()

    environment.run(keyboard_steering=False, ai_steering=True, **dict(ai_model=model))
    # test_show_obstacles()
