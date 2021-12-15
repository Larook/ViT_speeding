# import math
# import os
# import random
# import sys
import time

# import numpy as np
# import pybullet as pb
# import pybullet_data
# import torch
# import torchsummary as torchsummary
# from PIL import Image
# from matplotlib import pyplot as plt
#
# from Environment import Environment
from SimulationData import SimulationData
from ViTModelGenerator import ViTRegression
from ResnetModelGenerator import ResnetRegression

# import torch.onnx

import wandb
import pprint
from sweep_config import sweep_config

DETLA_T = 0.1

#TODO:
"""
- improve plots
    - for each plot add text message with the final training and testing loss
"""


def say_hi():
    print("hello from train")


def model_pipeline(hyperparameters):
    """ based on https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=IEUSDJByP_Yk """

    print("--------- hyperparameters ---------", hyperparameters)
    # training_pickle_df_path = 'model_training/data/03-11_all_training_data/whole_03-11_day_training_data.pkl'
    # training_pickle_df_path = 'model_training/data/whole_15-11_day_training_data.pkl'
    # training_pickle_df_path = 'model_training/data/whole_24-11_day_training_data.pkl'
    training_pickle_df_path = 'model_training/data/whole_25-11_day_training_data.pkl'

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
        wandb.config.early_stopping = hyperparameters['early_stopping']
        wandb.config.l2_regularization_weight = hyperparameters['l2_regularization_weight']

        num_outputs = 2
        # params = dict(image_size=256, patch_size=8, num_outputs=1, channels=3, dim=64, depth=1, heads=2, mlp_dim=128)
        params = dict(image_size=256, patch_size=8, num_outputs=num_outputs, channels=3, dim=64, depth=1, heads=2, mlp_dim=128)

        if wandb.config.model == 'ViT':
            model = ViTRegression(wandb_config=hyperparameters, **params)
        elif wandb.config.model == 'resnet':
            model = ResnetRegression(wandb_config=hyperparameters, num_outputs=num_outputs)
        model.load_dataloaders(pickle_df_path=training_pickle_df_path)

        model.train_epochs()
        model.plot_training_history()
    # model.to_onnx()
    # wandb.save("model.onnx")


def run_regular_wandb_training(model):

    wandb.login(key=open('wandb/__login_wandb_pwd.txt', 'r').read())
    wandb.init(project="my-test-project", entity="larook")
    # wandb.init(mode="disabled")


    time_start = time.time()
    config = dict(model=model, learning_rate=0.003, epochs=4, batch_size=10, optimizer='adam', early_stopping=True, l2_regularization_weight=0.1)
    # config = dict(model='resnet', learning_rate=0.003, epochs=1000, batch_size=10, optimizer='adam')
    model_pipeline(hyperparameters=config)
    print('GPU device time taken: ', time.time()-time_start)


def run_sweeps_wandb_training():
    def train_with_wandb_sweeps(config=None):
        with wandb.init(config=config):
            config = wandb.config  # load the config given by sweep agent
            print("hello from train")
            pprint.pprint(config)

            # params = dict(image_size=256, patch_size=8, num_outputs=1, channels=3, dim=64, depth=1, heads=2, mlp_dim=128)
            params = dict(image_size=256, patch_size=8, num_outputs=2, channels=3, dim=64, depth=1, heads=2, mlp_dim=128)
            model = ViTRegression(wandb_config=config, **params)

            # training_pickle_df_path = 'model_training/data/03-11_all_training_data/whole_03-11_day_training_data.pkl'
            # training_pickle_df_path = 'model_training/data/whole_15-11_day_training_data.pkl'
            # training_pickle_df_path = 'model_training/data/whole_24-11_day_training_data.pkl'
            training_pickle_df_path = 'model_training/data/whole_25-11_day_training_data.pkl'
            model.load_dataloaders(pickle_df_path=training_pickle_df_path)

            model.train_epochs()

    # check if sweep config is correctly loaded
    pprint.pprint(sweep_config)
    wandb.login(key=open('wandb/__login_wandb_pwd.txt', 'r').read())

    # start the sweeping project
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    # run the function using sweep agent
    wandb.agent(sweep_id, train_with_wandb_sweeps, count=100)  # count=how many times to run different settings


if __name__ == "__main__":
    # training_data_dir_path = 'model_training/data/03-11_10:23_training_data/'
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=r4VjKui20N3j

    # run_regular_wandb_training(model='ViT')
    # run_regular_wandb_training(model='resnet')
    run_sweeps_wandb_training()


