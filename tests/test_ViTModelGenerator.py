import unittest
from unittest import TestCase

from SimulationData import SimulationData
from ViTModelGenerator import ViTRegression


class TestViTRegression(TestCase):

    def setUp(self) -> None:
        self.dir_path = '../model_training/data/30-09_18:06_training_data/'

        self.model = ViTRegression(image_size=256, patch_size=8, num_outputs=1, channels=1,
                                   dim=64, depth=1, heads=2, mlp_dim=128)

        self.data = SimulationData(create=False)
        self.main_df = self.data.get_load_pickles_to_df(create=False, dir_path=self.dir_path)
        print("self.main_df", self.main_df.head())
        pass

    def tearDown(self) -> None:
        pass

    def test_load_simulation_data(self):
        self.model.load_simulation_data(self.main_df)
        print("self.model.outputs.size", self.model.outputs.size)
        # if self.model.outputs.size


    def test_forward(self):
        # self.fail()
        pass

    # def test_load_split_train_test(self):
    #     self.fail()
    #
    # def test_train_epoch(self):
    #     self.fail()
    #
    # def test_evaluate(self):
    #     self.fail()


if __name__ == '__main__':
    unittest.main()
