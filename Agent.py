import sys
import math
import numpy as np
import pybullet as pb
import pygame as pygame
from matplotlib import pyplot as plt


class Agent:
    id: int
    pose: list
    orientation: list
    velocity = 10
    steering_angle = 0
    step_angle = 0.2
    step_velocity = 0.2

    def __init__(self, urdf_path: str):
        self.pose = [0., 0., 0.1]
        self.orientation = [0, 0, math.pi / 2]
        agent_id = pb.loadURDF(urdf_path, self.pose, pb.getQuaternionFromEuler(self.orientation))
        self.id = agent_id
        pb.changeDynamics(bodyUniqueId=self.id,
                          linkIndex=self.id,
                          mass=1.1,  # this mass works with the box_example.urdf but doesnt with other boxes
                          lateralFriction=sys.maxsize,
                          spinningFriction=sys.maxsize,
                          rollingFriction=sys.maxsize,
                          restitution=0.0,
                          linearDamping=0.0,
                          angularDamping=0.0,
                          contactStiffness=-1,
                          contactDamping=-1)

        self.steering_angle_slider = pb.addUserDebugParameter("angle", -math.pi/2, math.pi/2, 0)
        self.velocity_slider = pb.addUserDebugParameter("v", 0, 30, 0)

    def angle_to_vy(self):
        # calculates the velocity of obstacles based on own velocity and the steering angle
        return math.cos(math.radians(self.steering_angle)) * self.velocity

    def take_image(self, display=False):
        res = 400
        # res = 256
        # target = self.pose + np.dot(self.orientation, [0, 0, 1.0, 1.0])[0:3]
        # print("target", target)
        # up = np.dot(self.orientation, [0, -1.0, 0, 1.0])[0:3]

        eye, orient = pb.getBasePositionAndOrientation(self.id)
        target = np.add(eye, [0, 100, 2])  # target = eye + np.matmul(orient, [0, 0, 10, 0])#[0:3]
        up = [0, 10, 10]

        view_matrix = pb.computeViewMatrix(eye, target, up)
        projection_matrix = pb.computeProjectionMatrixFOV(40, 1, 0.01, 100)

        img, _, _ = pb.getCameraImage(res, res, view_matrix, projection_matrix, shadow=1)[2:]
        if display:
            # this looks good
            plt.figure()
            plt.imshow(img)
            plt.title('captured img from agent')
            plt.show()

        return img[..., :3]

    def get_new_v(self, dt, keyboard=False):

        if not keyboard:
            self.steering_angle = pb.readUserDebugParameter(self.steering_angle_slider)
            self.velocity = pb.readUserDebugParameter(self.velocity_slider)
        else:
            keys = get_key_pressed()
            print("keys", keys)
            for key in keys:
                if key == "LEFT":
                    self.steering_angle = self.steering_angle - self.step_angle
                if key == "RIGHT":
                    self.steering_angle = self.steering_angle + self.step_angle
                if key == "UP":
                    self.velocity = self.velocity + self.step_velocity
                if key == "DOWN":
                    self.velocity = self.velocity - self.step_velocity

        v_y = self.angle_to_vy()
        return v_y


    def collision_detected(self):
        contact_points = pb.getContactPoints(bodyA=self.id)  # , physicsClientId=1)
        if len(contact_points) > 4:
            return True
        return False

    def update_pose(self, dt):
        pose, _ = pb.getBasePositionAndOrientation(self.id)
        pb.resetBasePositionAndOrientation(self.id, np.add(pose, [math.sin(self.steering_angle) * self.velocity * dt, 0, 0]),
                                           pb.getQuaternionFromEuler(self.orientation))
        pass


def get_key_pressed(relevant=None):
    pressed_keys = []
    code_to_name = {65297: 'UP', 65298: 'DOWN', 65296: 'RIGHT', 65295: 'LEFT'}
    events = pb.getKeyboardEvents()
    key_codes = events.keys()
    for key in key_codes:

        if key in code_to_name.keys():
            key = code_to_name[key]
        pressed_keys.append(key)
    return pressed_keys