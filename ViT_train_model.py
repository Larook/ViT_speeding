import math
import os
import random
import sys
import time

import numpy as np
import pybullet as pb
import pybullet_data
import torch
import torchsummary as torchsummary
from PIL import Image
from matplotlib import pyplot as plt

from Environment import Environment
from ViTModelGenerator import ViTRegression
from ResnetModelGenerator import ResnetRegression

import torch.onnx

import wandb
import pprint
from sweep_config import sweep_config

DETLA_T = 0.1

#TODO:
"""
- improve plots
    - for each plot add text message with the final training and testing loss
"""


def model_pipeline(hyperparameters):
    """ based on https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=IEUSDJByP_Yk """

    print("--------- hyperparameters ---------", hyperparameters)
    training_pickle_df_path = 'model_training/data/03-11_all_training_data/whole_03-11_day_training_data.pkl'

    # tell wandb to get started
    # with wandb.init(project="pytorch-demo_vit", config=hyperparameters):
    with wandb.init(project="pytorch-demo_vit"):
        # access all HPs through wandb.config, so logging matches execution!
        # config = dict(learning_rate=0.003, epochs=5, batch_size=10, optimizer='adam')

        wandb.config.learning_rate = hyperparameters['learning_rate']
        wandb.config.epochs = hyperparameters['epochs']
        wandb.config.batch_size = hyperparameters['batch_size']
        wandb.config.optimizer = hyperparameters['optimizer']
        wandb.config.model = hyperparameters['model']

        params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                      dim=64, depth=1, heads=2, mlp_dim=128)

        if wandb.config.model == 'ViT':
            model = ViTRegression(wandb_config=hyperparameters, **params)
        elif wandb.config.model == 'resnet':
            model = ResnetRegression(wandb_config=hyperparameters)
            # print('list(model.children())', list(model.children()))
            # exit(1)
        model.load_dataloaders(pickle_df_path=training_pickle_df_path)

        model.train_epochs()
        model.plot_training_history()
    model.to_onnx()
    wandb.save("model.onnx")


def run_regular_wandb_training():
    wandb.login(key='015c0cfe2e5003fb319b237200741f4647f204c3')
    wandb.init(project="my-test-project", entity="larook")


    time_start = time.time()
    # config = dict(model='ViT', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
    config = dict(model='resnet', learning_rate=0.003, epochs=20, batch_size=10, optimizer='adam')
    model_pipeline(hyperparameters=config)
    print('GPU device time taken: ', time.time()-time_start)


def run_sweeps_wandb_training():
    def train_with_wandb_sweeps(config=None):
        with wandb.init(config=config):
            config = wandb.config  # load the config given by sweep agent
            print("hello from train")
            pprint.pprint(config)

            params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                          dim=64, depth=1, heads=2, mlp_dim=128)
            model = ViTRegression(wandb_config=config, **params)

            # training_pickle_df_path = 'model_training/data/03-11_all_training_data/whole_03-11_day_training_data.pkl'
            training_pickle_df_path = 'model_training/data/whole_15-11_day_training_data.pkl'
            model.load_dataloaders(pickle_df_path=training_pickle_df_path)

            model.train_epochs()

    # check if sweep config is correctly loaded
    pprint.pprint(sweep_config)
    wandb.login(key=open('wandb/__login_wandb_pwd.txt', 'r').read())

    # start the sweeping project
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    # run the function using sweep agent
    wandb.agent(sweep_id, train_with_wandb_sweeps, count=40)  # count=how many times to run different settings


if __name__ == "__main__":
    # training_data_dir_path = 'model_training/data/03-11_10:23_training_data/'
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=r4VjKui20N3j
    run_regular_wandb_training()
    # run_sweeps_wandb_training()


