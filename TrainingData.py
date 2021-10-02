import csv
import os
import time
from datetime import date, datetime
import pickle

import pandas as pd


class TrainingData():
    """ save everything to pandas df to easily save to csv
        make sure that not everything is saved in RAM
        if collision detected discard last X seconds of data to save """
    rows_max = 100

    def __init__(self):
        self.start_time = time.time()
        self.rows_in_memory = 0
        self.memory = []
        self.pickle_part = 0

        # create a csv with the month_day_h_m
        now = datetime.now()
        dt_string = now.strftime("%d-%m_%H:%M")
        self.dir_path = 'model_training/data/' + dt_string + '_training_data'
        # print("os.path.exists(self.dir_path)", os.path.exists(self.dir_path))
        if 'tests' not in os.getcwd():
            if not os.path.exists(self.dir_path):
                os.mkdir(self.dir_path)

        self.filepath = self.dir_path + dt_string

        # with open(self.filepath, 'w', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     # write the header
        #     header = ['time', 'steering_angle', 'velocity_y', 'image']
        #     writer.writerow(header)
        pass

    def save_training_information(self, img, angle, v_y):
        row = dict(time=time.time()-self.start_time, steering_angle=angle, velocity_y=v_y, image=img)
        self.memory.append(row)
        self.rows_in_memory += 1

        if self.rows_in_memory >= self.rows_max:
            # flush the memory to save the existing csv
            df = pd.DataFrame(self.memory)

            df.to_pickle(self.filepath + "_p" + str(self.pickle_part) + ".pkl")
            self.memory = []
            self.pickle_part += 1

        pass

    def get_load_pickles_to_df(self):
        # go through the directory with pickles as parts and concat the dfs to get the full df
        # print("\n\nself.dir_path", self.dir_path)

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


