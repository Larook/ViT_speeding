import os
import pandas as pd
from PIL import Image

from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class SimulationImageDataset(Dataset):
    def __init__(self, main_df, transform=None, target_transform=None):
        self.df = main_df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = self.df['image'][idx]
        # label = self.img_labels.iloc[idx, 1]
        steering_angle = self.df['steering_angle'][idx]
        velocity_y = self.df['velocity_y'][idx]
        if self.transform:
            img = Image.fromarray(image, 'RGB')
            image = self.transform(img)
        if self.target_transform:
            steering_angle = self.target_transform(steering_angle)
            velocity_y = self.target_transform(velocity_y)
        return image, steering_angle, velocity_y

    def get_training_test_split(self, training_percentage):
        # shuffle df
        df = self.df.sample(frac=1)

        # get the index of 80%
        max_training_idx = int(training_percentage * len(df))

        # return the lists
        imgs_training = df['image'][:max_training_idx]
        angles_training = df['steering_angle'][:max_training_idx]
        vels_training = df['velocity_y'][:max_training_idx]

        imgs_test = df['image'][max_training_idx:]
        angles_test = df['steering_angle'][max_training_idx:]
        vels_test = df['velocity_y'][max_training_idx:]

        return imgs_training, angles_training, vels_training, imgs_test, angles_test, vels_test

