import unittest
from unittest import TestCase

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from SimulationData import SimulationData
from SimulationImageDataset import SimulationImageDataset
from ViTModelGenerator import ViTRegression

# TODO:
"""
[X] - fix forward()
[X] - checkout training in batches -> works
[X] - add saving of the model
[ ] - use model to make an action
[ ] - add schedulers
[ ] - look into 'tensorvision'
"""


class TestViTRegression(TestCase):

    def setUp(self) -> None:
        self.dir_path = '../model_training/data/30-09_18:06_training_data/'

        self.params = dict(image_size=256, patch_size=8, num_outputs=1, channels=1,
                              dim=64, depth=1, heads=2, mlp_dim=128)

        model = ViTRegression(**self.params)

        model.load_dataloaders(dir_path=self.dir_path)
        pass

    def tearDown(self) -> None:
        pass

    def test_forward(self):
        model = ViTRegression(**self.params)

        model.load_dataloaders(dir_path=self.dir_path)
        model.eval()

        for i, [imgs, angles, vels] in enumerate(model.train_dataloader):
            output = model.forward(imgs)
            print("output", output)
            break

        pass

    def test_plot_training_history(self):
        model = ViTRegression(**self.params)
        model.load_dataloaders(dir_path=self.dir_path)

        model.loss_train_history = [0.2, 0.1, 0.05]
        model.loss_test_history = [0.25, 0.15, 0.01]
        model.plot_training_history()
        pass

    def test_train_epoch(self):
        model = ViTRegression(**self.params)
        model.load_dataloaders(dir_path=self.dir_path)
        model.train_epochs(max_epochs=50, save_path='test_model.pth')
        model.plot_training_history()
        pass


    def test_load_model(self):
        model = ViTRegression(**self.params)
        model.load_state_dict(torch.load('test_model_60epochs.pth'))
        model.load_dataloaders(dir_path=self.dir_path)
        model.eval()

        for i, [imgs, angles, vels] in enumerate(model.test_dataloader):
            output = model.forward(imgs)
            print("output", output)
            print("angles", angles)

            output = model.forward(imgs)
            target = angles.unsqueeze(1).float()
            loss = F.smooth_l1_loss(output, target)  # L1 loss for regression applications
            # loss.backward()
            # self.optimizer.step()
            print("loss.item():", loss.item())

            break


if __name__ == '__main__':
    unittest.main()
