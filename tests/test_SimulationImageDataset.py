import unittest
from unittest import TestCase

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from SimulationData import SimulationData
from SimulationImageDataset import SimulationImageDataset


class TestSimulationImageDataset(TestCase):

    def startUp(self):
        self.dir_path = '../model_training/data/30-09_18:06_training_data/'
        data = SimulationData(create=False)
        main_df = data.get_load_pickles_to_df(create=False, dir_path=self.dir_path)
        print("main_df", main_df)
        pass

    def tearDown(self):
        pass

    def test_show_images_from_df(self):
        dir_path = '../model_training/data/30-09_18:06_training_data/'
        data = SimulationData(create=False)
        main_df = data.get_load_pickles_to_df(create=False, dir_path=dir_path)
        dataset = SimulationImageDataset(main_df)
        img, angle, vel = dataset[0]

        print('angle', angle, 'vel', vel)
        image = Image.fromarray(img, 'RGB')
        image.show()
        pass

    def test_get_training_test_split(self):
        dir_path = '../model_training/data/30-09_18:06_training_data/'
        data = SimulationData(create=False)
        main_df = data.get_load_pickles_to_df(create=False, dir_path=dir_path)
        dataset = SimulationImageDataset(main_df)

        imgs_training, angles_training, vels_training, imgs_test, angles_test, vels_test = dataset.get_training_test_split(0.8)
        self.assertEqual(len(imgs_training) + len(imgs_test), len(main_df))
        pass


    def test_dataloaders(self):
        from torch.utils.data import DataLoader

        dir_path = '../model_training/data/30-09_18:06_training_data/'
        data = SimulationData(create=False)

        # splits main_df to TRAIN and TEST, because we need separate trainloaders
        train_df, test_df = data.load_dfs_from_pickles(create=False, training_percentage=0.8, dir_path=dir_path, shuffle=True)

        # split the df into training and testing
        for df in [train_df, test_df]:
            dataset_train = SimulationImageDataset(main_df=df)

            # create dataloaders
            train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)

            # Display image and label.
            train_imgs, train_angles, train_vels = next(iter(train_dataloader))
            # returned values are batches of many objects!

            train_imgs = train_imgs.detach().numpy()
            print('train_angles[0]', train_angles[0], 'train_vels[0]', train_vels[0], 'train_imgs[0]', train_imgs[0])
            image = Image.fromarray(train_imgs[0], 'RGB')
            image.show()



if __name__ == '__main__':
    unittest.main()
