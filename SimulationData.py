import csv
import os
import time
from datetime import date, datetime
import pickle

import pandas as pd


class SimulationData():
    """ save everything to pandas df to easily save to csv
        make sure that not everything is saved in RAM
        if collision detected discard last X seconds of data to save """
    rows_max = 20

    def __init__(self, create=True):
        self.start_time = time.time()
        self.rows_in_memory = 0
        self.memory = []
        self.pickle_part = 0

        # create a csv with the month_day_h_m
        if create:
            now = datetime.now()
            dt_string = now.strftime("%d-%m_%H:%M")
            self.dir_path = 'model_training/data/' + dt_string + '_training_data'
            # print("os.path.exists(self.dir_path)", os.path.exists(self.dir_path))
            if 'tests' not in os.getcwd():
                if not os.path.exists(self.dir_path):
                    os.mkdir(self.dir_path)
            self.filepath = self.dir_path + '/' + dt_string

    def save_training_information(self, img, angle, v_y):
        row = dict(time=time.time()-self.start_time, steering_angle=angle, velocity_y=v_y, image=img)
        self.memory.append(row)
        self.rows_in_memory += 1
        print("<> save_training_information ->", self.filepath + "_p" + str(self.pickle_part) + ".pkl")
        if self.rows_in_memory >= self.rows_max:
            # flush the memory to save the existing csv
            df = pd.DataFrame(self.memory)

            df.to_pickle(self.filepath + "_p" + str(self.pickle_part) + ".pkl")
            self.memory = []
            self.pickle_part += 1

        pass

    def get_load_pickles_to_df(self, create=True, **kwargs):
        # go through the directory with pickles as parts and concat the dfs to get the full df
        # print("\n\nself.dir_path", self.dir_path)

        if not create:
            self.dir_path = kwargs['dir_path']
            print("self.dir_path =", self.dir_path)

        for i, file in enumerate(os.listdir(self.dir_path)):
            # print("file", file)
            if i == 0:
                main_df = pd.read_pickle(self.dir_path + file)
            else:
                df = pd.read_pickle(self.dir_path + file)
                # print("df.columns", df.columns)
                main_df = main_df.append(df, ignore_index=True)
                # print("main_df.size", main_df.size)
        # print("end of files")

        return main_df

    def load_dfs_from_pickles(self, create, training_percentage, dir_path, shuffle):
        # load the pickles to df
        df = self.get_load_pickles_to_df(create, dir_path=dir_path)

        # shuffle df
        if shuffle:
            df.sample(frac=1).reset_index(drop=True)

        # get the index of 80%
        max_training_idx = int(training_percentage * len(df))

        train_df = df[:max_training_idx]
        test_df = df[max_training_idx:]

        if shuffle:
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

        return train_df, test_df

