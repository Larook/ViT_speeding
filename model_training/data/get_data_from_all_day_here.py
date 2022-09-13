from SimulationDataLoader import SimulationDataLoader


if __name__ == "__main__":
    dir_path = '.'
    # dir_name_begin_day = '15-11_'
    # dir_name_begin_day = '24-11_'  # multicolor 76 imgs
    # dir_name_begin_day = '22-12'  # white 976 imgs
    dir_name_begin_day = 'SmallWhite_22-12'  # white 67 imgs
    data = SimulationDataLoader(create=False)
    main_df = data.get_loaded_pickles_one_day(create=False, dir_path=dir_path, dir_name_begin=dir_name_begin_day)
    print("main_df", main_df)

    main_df.to_pickle("whole_" + dir_name_begin_day + "day_training_data.pkl", protocol=4)
    main_df.to_csv("whole_" + dir_name_begin_day + "day_training_data.csv")
