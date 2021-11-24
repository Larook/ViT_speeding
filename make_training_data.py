from SimulationData import SimulationData

def make_training_dataset():
    """ for running in the colab """
    dir_path = 'model_training/data/'
    dir_name_begin_day = '15-11_'
    data = SimulationData(create=False)
    main_df = data.get_load_pickles_of_one_day_to_df(create=False, dir_path=dir_path, dir_name_begin=dir_name_begin_day)
    print("main_df", main_df)
    main_df.to_pickle("whole_" + dir_name_begin_day + "day_training_data.pkl")