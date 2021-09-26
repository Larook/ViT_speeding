import os

from Agent import Agent

import math
import numpy as np
import pybullet as pb
import pybullet_data


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
        self.sim_dt = dt
        pb.setTimeStep(self.sim_dt)
        pb.setGravity(0, 0, -9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = pb.loadURDF('plane.urdf')

        plane_id = pb.loadURDF("urdfs/surroundings/ground.urdf", basePosition=[0, 3, 0], useFixedBase=1)
        background_id = pb.loadURDF("urdfs/surroundings/background.urdf", basePosition=[0, 8, 0],
                                    baseOrientation=pb.getQuaternionFromEuler((math.pi/2, math.pi/2, 0)),
                                    useFixedBase=1)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # xin = pb.addUserDebugParameter("x", -0.224, 0.224, 0)
        # yin = pb.addUserDebugParameter("y", -0.224, 0.224, 0)
        # zin = pb.addUserDebugParameter("z", 0, 1., 0.5)

    def get_random_hor_pose(self):
        return np.add(self.horizon_middle_point, [np.random.uniform(-2.4, 2.4), np.random.uniform(0.4, 12), 0])

    def spawn_agent(self):
        self.agent = Agent(urdf_path="urdfs/agent.urdf")
        self.agent_id = self.agent.id

    def run(self):
        """ whole procedure of creating a simulation
        moves objects, resets their position and makes sure that agent saves images"""

        def spawn_random_obstacle():
            # spawns random obstacle from directory at the horizon
            # todo: How to make sure that objects disappear and appear with the correct distance inbetween?
            # todo: size of cars and size of road - how many lanes?
            for i, urdf in enumerate(os.listdir(os.getcwd() + '/urdfs/obstacles/')):
                print("urdf=", urdf)
                obs_id = pb.loadURDF(os.getcwd() + '/urdfs/obstacles/' + urdf, self.get_random_hor_pose(), self.obst_quat)
                self.randomize_color(obs_id)

                self.env_pb_obstacle_ids.append(obs_id)
            pass

        spawn_random_obstacle()
        enough_data_created = False

        # pb.setRealTimeSimulation(1)
        while 1:  # not enough_data_created:
            idx_to_reset = self.move_and_get_obstacle_idx_to_reset()
            self.reset_obstacles(idx_to_reset)

            img = self.agent.take_image(display=False)
            pb.stepSimulation()


        print("Simulation stopped")
        return None

    def move_and_get_obstacle_idx_to_reset(self) -> list:
        idx_to_remove = []

        # p_x, v_y = self.agent.get_updated_dynamics(self.sim_dt)
        v_y = self.agent.get_updated_dynamics(self.sim_dt, keyboard=True)
        print("steering_angle=%f  v_Y=%f" % (self.agent.steering_angle, v_y))
        # print("v_y", v_y)

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

