import os
import unittest
from unittest import TestCase

import pandas as pd

from TrainingData import TrainingData


class TestTrainingData(unittest.TestCase):
    def setUp(self):
        print("hello from test")

    def tearDown(self):
        print("hello from teardown")

    def test_get_load_pickles_to_df(self):
        print("hello from test")
        print("os.getcwd()", os.getcwd())

        data = TrainingData()
        data.dir_path = '../model_training/data/30-09_18:06_training_data/'
        main_df = data.get_load_pickles_to_df()
        self.assertEqual(type(main_df), pd.DataFrame)
        print("managed to load big df")


if __name__ == '__main__':
    unittest.main()
