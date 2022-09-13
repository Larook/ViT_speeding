import math
import os
import random
import sys
import time

import numpy as np
import pybullet as pb
import pybullet_data
from PIL import Image
from matplotlib import pyplot as plt

from Environment import Environment

DETLA_T = 0.1

#TODO:
"""
OK - end simulation when collision detected
- setup the environment with 4 lanes of cars
    OK- 4 different x positions for cars to spawn
    OK- width of the cars / lanes
    OK- field of view of the camera
- what about the velocity? When playing try to increase it sometimes
    - Add saving the pictures (if collided dont save last 2 seconds of decisions)
        - it seems like saving csvs doesnt work:
            OK -try to instead save the dataframes to pickles and then load full df as pickle and check if loading image works
            OK - make a TrainingData method that loads all of the small pickles into a big pickle with big DF.
"""

if __name__ == "__main__":
    # define environment
    # difficulty_distance = 12
    difficulty_distance = 22
    ai_steering = False

    environment = Environment(dt=DETLA_T, ai_steering=ai_steering, difficulty_distance=difficulty_distance)
    # img = environment.agent.get_image_from_camera(display=True)
    # time.sleep(1)
    environment.set_cool_game_vibe_camera_position()

    environment.run(keyboard_steering=True, ai_steering=ai_steering)
