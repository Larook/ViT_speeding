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

if __name__ == "__main__":
    # define environment
    environment = Environment(dt=DETLA_T)

    environment.spawn_agent()
    environment.run()

    # test_show_obstacles()
