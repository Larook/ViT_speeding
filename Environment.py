import os
import random
import time

import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import transforms
import PySimpleGUI as sg

from Agent import Agent

import math
import numpy as np
import pybullet as pb
import pybullet_data

from SimulationData import SimulationData


class Environment():
    # when clicking "G" when sim window pops out, the disturbing windows dissapear
    sim_dt: int
    agent: Agent
    env_pb_obstacle_ids = []

    horizon_middle_point = [0, 8.4, 0.2]
    obst_quat = pb.getQuaternionFromEuler([0, 0, math.pi/2])

    dist_per_step = 0.0005

    def __init__(self, dt):
        physicsClient = pb.connect(pb.GUI)
        # physicsClient = pb.connect(pb.UDP, "192.168.0.10")
        # physicsClient = pb.connect(pb.TCP, "localhost", 6667)
        while not pb.isConnected():
            pass
        self.sim_dt = dt
        pb.setTimeStep(self.sim_dt)
        pb.setGravity(0, 0, -9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = pb.loadURDF('plane.urdf')

        plane_id = pb.loadURDF("urdfs/surroundings/ground.urdf", basePosition=[0, 3, 0], useFixedBase=1)
        background_id = pb.loadURDF("urdfs/surroundings/background.urdf", basePosition=[0, 20, 0],
                                    baseOrientation=pb.getQuaternionFromEuler((math.pi/2, math.pi/2, 0)),
                                    useFixedBase=1)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # xin = pb.addUserDebugParameter("x", -0.224, 0.224, 0)
        # yin = pb.addUserDebugParameter("y", -0.224, 0.224, 0)
        # zin = pb.addUserDebugParameter("z", 0, 1., 0.5)
        self.training_data = SimulationData()

    def get_random_hor_pose(self):
        # make sure that the positions are within ranges for each line
        car_width = 0.7
        lane_xs = [-1.5, -0.5, 0.5, 1.5]
        return np.add(self.horizon_middle_point, [np.random.choice(lane_xs), np.random.uniform(0.4, 60), 0])

    def spawn_agent(self):
        self.agent = Agent(urdf_path="urdfs/agent.urdf")
        self.agent_id = self.agent.id

    def run(self, keyboard_steering, ai_steering, **kwargs):
        """ whole procedure of creating a simulation
        moves objects, resets their position and makes sure that agent saves images"""

        def spawn_random_obstacles():
            # spawns random obstacle from directory at the horizon
            for i, urdf in enumerate(os.listdir(os.getcwd() + '/urdfs/obstacles/')):
                print("urdf=", urdf)
                obs_id = pb.loadURDF(os.getcwd() + '/urdfs/obstacles/' + urdf, self.get_random_hor_pose(), self.obst_quat)
                self.randomize_color(obs_id)

                self.env_pb_obstacle_ids.append(obs_id)
            pass

        spawn_random_obstacles()

        # print("kwargs['ai_model']", kwargs['ai_model'])

        if keyboard_steering:
            self.gather_data()
        if ai_steering:
            self.evaluate_ai(kwargs['ai_model'])

        print("Simulation stopped")
        return None

    def move_and_get_obstacle_idx_to_reset(self, v_y :float) -> list:
        idx_to_remove = []

        for i in self.env_pb_obstacle_ids:
            pb.resetBaseVelocity(i, linearVelocity=[0, -v_y, 0], angularVelocity=[0, 0, 0])
            pos, rot = pb.getBasePositionAndOrientation(i)
            if pos[1] < -1:
                idx_to_remove.append(i)
        return idx_to_remove

    def reset_obstacles(self, idx_to_reset):
        for i in idx_to_reset:
            print("resets the position")
            self.randomize_color(i)
            pb.resetBasePositionAndOrientation(i, self.get_random_hor_pose(), self.obst_quat)
        pass

    def set_obstacles_velocity(self):
        for i in self.env_pb_obstacle_ids:
            pb.resetBaseVelocity(i, linearVelocity=[0, -self.agent.velocity, 0], angularVelocity=[0, 0, 0])

    def randomize_color(self, obs_id):
        # doesnt work! Dont know why
        for joint in range(10):
            pb.changeVisualShape(obs_id, joint, rgbaColor=[np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), 1], physicsClientId=0)
        return None

    def gather_data(self):
        """ run the simulaton and save data """
        enough_data_created = False

        can_proceed = True
        while can_proceed:  # not enough_data_created:
            # if agent in collision then stop the game
            can_proceed = not self.agent.collision_detected() and self.agent.is_on_the_lane()

            # take and save the image
            img = self.agent.take_image(display=False)

            # save the action to take for current img
            v_y = self.agent.get_new_v(self.sim_dt, keyboard=True)

            # move the agent
            self.agent.update_pose(dt=self.sim_dt)
            print("steering_angle=%f  v_Y=%f" % (self.agent.steering_angle, v_y))

            # save img-angle-vel_y
            self.training_data.save_training_information(img, self.agent.steering_angle, v_y)

            # move all the obstacles and reset their positions when needed
            idx_to_reset = self.move_and_get_obstacle_idx_to_reset(v_y)
            self.reset_obstacles(idx_to_reset)

            pb.stepSimulation()

        save_the_trajectories = self.popup_window_ask_if_save()
        if not save_the_trajectories:
            # delete the folder
            print("self.dir_path", self.training_data.dir_path)
            files_in_dir = os.listdir(self.training_data.dir_path)
            print("files_in_dir", files_in_dir)
            os.rmdir(self.training_data.dir_path)
            print("REMOVED ALL FILES FROM THIS ATTEMPT")


    def evaluate_ai(self, ai_model):
        print("*********************** evaluate_ai ***********************")
        preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.Grayscale(num_output_channels=1),
                                               # transforms.RandomVerticalFlip(),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               ])
        can_proceed = True
        # print("self.agent.is_on_the_lane()", self.agent.is_on_the_lane())
        while can_proceed:
            can_proceed = not self.agent.collision_detected() and self.agent.is_on_the_lane()

            # if agent in collision then stop the game
            img = self.agent.take_image(display=False)

            # need to change dimensions of img (assume batch = 1)
            input_tensor = preprocess(Image.fromarray(img))
            input_batch = input_tensor.unsqueeze(0)

            angle = ai_model.forward(input_batch)
            self.agent.steering_angle = angle

            # move the agent
            self.agent.update_pose(dt=self.sim_dt)

            v_y = self.agent.angle_to_vy()
            print("steering_angle=%f  v_Y=%f" % (self.agent.steering_angle, v_y))
            # move all the obstacles and reset their positions when needed
            idx_to_reset = self.move_and_get_obstacle_idx_to_reset(v_y)
            self.reset_obstacles(idx_to_reset)

            pb.stepSimulation()
        pass

    def popup_window_ask_if_save(self):
        answer = sg.popup_yes_no('Do you want to save this approach?')
        if answer == 'No':
            sg.popup_cancel('Deleting the trajectories')
            return False
        elif answer == "Yes":
            sg.popup_cancel("Ok saving the trajectories")
            return True

