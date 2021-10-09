import os
import unittest
from unittest import TestCase

import pandas as pd

from SimulationData import SimulationData


class TestTrainingData(unittest.TestCase):
    def setUp(self):
        print("hello from test")
        self.data = SimulationData()


    def tearDown(self):
        print("hello from teardown")

    def test_get_load_pickles_to_df(self):
        print("hello from test")
        print("os.getcwd()", os.getcwd())

        self.data.dir_path = '../model_training/data/30-09_18:06_training_data/'
        main_df = self.data.get_load_pickles_to_df()
        self.assertEqual(type(main_df), pd.DataFrame)
        print("managed to load big df")


if __name__ == '__main__':
    unittest.main()
