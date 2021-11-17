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
[ ] - How to use RGB images? For now it is running on 1 channel - Grayscale
[ ] - How to exclude the Agent going off the road?
    - make leaving the lanes same condition as hitting an obstacle
    - rerecord training data
[ ] - How to use the sequential data?
    - reread the paper
[ ] - How to make comparisons? How do we know if this method is better than sth other?
"""

if __name__ == "__main__":

    config = dict(learning_rate=0.003, epochs=30, batch_size=10, optimizer='adam')
    params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                  dim=64, depth=1, heads=2, mlp_dim=128)

    model = ViTRegression(wandb_config=config, **params)
    # model.load_state_dict(torch.load('model_training/trained_models/model_100_data_3-11.pth'))
    model.load_state_dict(torch.load('model_training/trained_models/model_100.pth'))
    model.eval()

    # define environment
    ai_steering = True
    difficulty_distance = 30
    environment = Environment(dt=DETLA_T, ai_steering=ai_steering, difficulty_distance=difficulty_distance)
    environment.set_cool_game_vibe_camera_position()

    environment.run(keyboard_steering=False, ai_steering=True, **dict(ai_model=model))
    # test_show_obstacles()
