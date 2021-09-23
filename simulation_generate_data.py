import math
import os
import random
import sys
import time

import numpy as np
import pybullet as pb
import pybullet_data
from PIL import Image


def load_buddy_urdf() -> int:
    visualShapeId = pb.createVisualShape(
        shapeType=pb.GEOM_MESH,
        fileName='urdfs/Buddy.obj',
        rgbaColor=None,
        meshScale=[0.1, 0.1, 0.1])

    collisionShapeId = pb.createCollisionShape(
        shapeType=pb.GEOM_MESH,
        fileName='urdfs/Buddy.obj',
        meshScale=[0.1, 0.1, 0.1])

    buddy_id = pb.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=[0, 0, 0.2],
        baseOrientation=pb.getQuaternionFromEuler([math.pi / 2, 0, 0]))
    return buddy_id

def load_car_urdf() -> int:
    visualShapeId = pb.createVisualShape(
        shapeType=pb.GEOM_MESH,
        fileName='urdfs/Buddy.obj',
        rgbaColor=None,
        meshScale=[0.1, 0.1, 0.1])

    collisionShapeId = pb.createCollisionShape(
        shapeType=pb.GEOM_MESH,
        fileName='urdfs/Buddy.obj',
        meshScale=[0.1, 0.1, 0.1])

    buddy_id = pb.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=[0, 0, 0.2],
        baseOrientation=pb.getQuaternionFromEuler([math.pi / 2, 0, 0]))
    return buddy_id


def get_pos_pb_object(pb_id: int) -> list:
    pos, _ = pb.getBasePositionAndOrientation(pb_id)
    return pos


class Agent:
    id: int
    pose: list
    orientation: list

    def __init__(self, urdf_path: str):
        self.pose = [0., 0., 0.3]
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


    def take_image(self):
        res = 400
        # target = self.pose + np.dot(self.orientation, [0, 0, 1.0, 1.0])[0:3]
        # print("target", target)
        # up = np.dot(self.orientation, [0, -1.0, 0, 1.0])[0:3]

        eye, orient = pb.getBasePositionAndOrientation(self.id)
        target = np.add(eye, [0, 100, 0])  # target = eye + np.matmul(orient, [0, 0, 10, 0])#[0:3]
        up = [0, 0, 2]

        view_matrix = pb.computeViewMatrix(eye, target, up)
        projection_matrix = pb.computeProjectionMatrixFOV(30, 1, 0.001, 10)

        img, _, _ = pb.getCameraImage(res, res, view_matrix, projection_matrix, shadow=1)[2:]
        return img[..., :3]


class Environment():
    # when clicking "G" when sim window pops out, the disturbing windows dissapear
    sim_dt: int
    agent: Agent
    env_pb_obstacle_ids = []

    horizon_middle_point = [0, 8, 0]
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
        return np.add(self.horizon_middle_point, [np.random.uniform(-2.4, 2.4), 0.4, 0])

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
            for urdf in os.listdir(os.getcwd() + '/urdfs/obstacles/'):
                print("urdf=", urdf)
                obs_id = pb.loadURDF(os.getcwd() + '/urdfs/obstacles/' + urdf, self.get_random_hor_pose(), self.obst_quat)
                self.env_pb_obstacle_ids.append(obs_id)
            pass

        spawn_random_obstacle()
        enough_data_created = False
        while 10000:  # not enough_data_created:
            idx_to_reset = self.move_obstacles_get_idx_to_reset()
            self.reset_obstacles(idx_to_reset)

            # todo: take and display image
            image = self.agent.take_image()
            # # from PIL import Image
            # # import numpy as np
            # img = Image.fromarray(image, 'RGB')
            # img.show()
            pass

        print("Simulation stopped")
        return None

    def move_obstacles_get_idx_to_reset(self) -> list:
        idx_to_remove = []

        for i in self.env_pb_obstacle_ids:
            pos, rot = pb.getBasePositionAndOrientation(i)
            new_pos = np.add(pos, [0, -self.dist_per_step, 0])
            pb.resetBasePositionAndOrientation(i, new_pos, rot)

            # get indexes to reset
            if new_pos[1] < -1:
                idx_to_remove.append(i)
        return idx_to_remove

    def reset_obstacles(self, idx_to_reset):
        for i in idx_to_reset:
            pb.resetBasePositionAndOrientation(i, self.get_random_hor_pose(), self.obst_quat)
        pass





def test_show_obstacles():
    # buddy_id = load_buddy_urdf()

    agent = Agent(urdf_path="urdfs/agent.urdf")
    # background_id.get_pose
    p, orient = pb.getBasePositionAndOrientation(agent.id)
    print("pos=", p, "orient=", orient)

    obs_1_id = pb.loadURDF("urdfs/obstacles/obstacle_1.urdf", [0., 1., 0.1], pb.getQuaternionFromEuler((0, 0, 0)))
    obs_1_id = pb.loadURDF("urdfs/obstacles/obstacle_1.urdf", [0., 4., 0.1], pb.getQuaternionFromEuler((0, 0, 0)))
    obs_1_id = pb.loadURDF("urdfs/obstacles/obstacle_1.urdf", [0., 8., 0.1], pb.getQuaternionFromEuler((0, 0, 0)))
    obs_2_id = pb.loadURDF("urdfs/obstacles/obstacle_2.urdf", [0, -1., 0.1], pb.getQuaternionFromEuler((0, 0, 0)))
    obs_2_id = pb.loadURDF("urdfs/obstacles/obstacle_3.urdf", [-1., -1., 0.1], pb.getQuaternionFromEuler((0, 0, 0)))

    for i in range(10000):
        pos, rot = pb.getBasePositionAndOrientation(agent.id)
        pb.resetBasePositionAndOrientation(agent.id, posObj=np.add(pos, [0.2 * math.cos(i/10), 0 , 0]), ornObj=rot)
        agent.take_image()
        pb.stepSimulation()
        time.sleep(0.005)



if __name__ == "__main__":
    # define environment
    environment = Environment(dt=0.1)

    environment.spawn_agent()
    environment.run()

    # test_show_obstacles()
