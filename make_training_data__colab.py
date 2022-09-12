#!/usr/bin/env python3
import os.path
from SimulationDataLoader import SimulationDataLoader

def make_training_dataset():
    """ for running in the colab """
    dir_path = 'model_training/data/'
    # dir_name_begin_day = '15-11_'
    # dir_name_begin_day = '24-11_'
    dir_name_begin_day = '25-11_'

    data = SimulationDataLoader(create=False)

    # !pip3 install pickle5
    # import pickle5 as pickle

    main_df = data.get_loaded_pickles_one_day(create=False, dir_path=dir_path, dir_name_begin=dir_name_begin_day)
    print("main_df", main_df)
    file_name = "whole_" + dir_name_begin_day + "day_training_data.pkl"
    file_path = os.path.join(os.getcwd(), 'model_training/data', file_name)
    main_df.to_pickle(file_path, protocol=4)


if __name__ == "__main__":
    make_training_dataset()