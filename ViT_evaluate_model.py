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

    num_outputs = 2
    if do_vit:
        params = dict(image_size=256, patch_size=8, num_outputs=num_outputs, channels=3,
                      dim=64, depth=1, heads=2, mlp_dim=128)
        # config_ViT = dict(model='ViT', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
        # config_ViT = {'batch_size': 10, 'dropout': 0.3, 'early_stopping': False, 'epochs': 200, 'l2_regularization_weight': 0, 'learning_rate': 0.0322595974979814, 'model': 'ViT', 'optimizer': 'sgd'}
        # config_ViT = dict(model='ViT', learning_rate=0.003, epochs=40, batch_size=10, optimizer='adam', early_stopping=True, l2_regularization_weight=0.1)


        # config_ViT = dict(model='ViT', learning_rate=0.003, epochs=1000, batch_size=10, optimizer='adam', early_stopping=True, l2_regularization_weight=0.1)
        # model = ViTRegression(wandb_config=config_ViT, **params)
        # model.load_state_dict(torch.load('model_training/trained_models/model_56_ViT_1000_early_stopping.pth'))


        # model.load_state_dict(torch.load('model_training/trained_models/model_700_ViT_data_24th.pth'))
        # model.load_state_dict(torch.load('model_training/trained_models/model_1000_ViT.pth'))

        # model.load_state_dict(torch.load('model_training/trained_models/model_200_ViT.pth'))
        # model.load_state_dict(torch.load('model_training/trained_models/model_40_ViT.pth'))


        # verdict - get back to only 1 output so we teach only steering angle
        config_ViT = dict(model='ViT', learning_rate=0.05881, epochs=1000, batch_size=10, optimizer='adam', early_stopping=False, l2_regularization_weight=0.1)
        model = ViTRegression(wandb_config=config_ViT, **params)
        # model.load_state_dict(torch.load('model_training/trained_models/model_1000_ViT_from_sweeps_0.pth'))
        model.load_state_dict(torch.load('model_training/trained_models/model_56_ViT_1000_early_stopping.pth'))

    else:
        # config_resnet = dict(model='resnet', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
        # model = ResnetRegression(wandb_config=config_resnet, num_outputs=num_outputs)
        # model.load_state_dict(torch.load('model_training/trained_models/model_20_resnet.pth'))

        config_resnet = {'batch_size': 10, 'dropout': 0.4, 'early_stopping': False, 'epochs': 200, 'l2_regularization_weight': 0.3,
         'learning_rate': 0.07573456017600638, 'model': 'resnet', 'optimizer': 'sgd'}
        model = ResnetRegression(wandb_config=config_resnet, num_outputs=num_outputs)
        model.load_state_dict(torch.load('model_training/trained_models/model_200_resnet.pth'))

    # define environment
    ai_steering = True
    difficulty_distance = 30
    environment = Environment(dt=DETLA_T, ai_steering=ai_steering, difficulty_distance=difficulty_distance, connect_gui=False)
    environment.set_cool_game_vibe_camera_position()

    # environment.run(eval_run=True, keyboard_steering=False, ai_steering=True, **dict(ai_model=model))
    # environment.run(eval_run=False, keyboard_steering=False, ai_steering=True, **dict(ai_model=model))

    environment.evaluate_many_tries(repeat_times=50, model=model)

    # test_show_obstacles()
