import math
import sys
import time

import numpy as np
import pybullet as pb
# import pybullet_data
import pybullet_data


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



class Environment():
    # when clicking "G" when sim window pops out, the disturbing windows dissapear
    env_pb_objects = []
    def __init__(self, dt):
        physicsClient = pb.connect(pb.GUI)
        pb.setTimeStep(dt)
        pb.setGravity(0, 0, -9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = pb.loadURDF('plane.urdf')

        plane_id = pb.loadURDF("urdfs/surroundings/ground.urdf", basePosition=[0, 3, 0], useFixedBase=1)
        background_id = pb.loadURDF("urdfs/surroundings/background.urdf", basePosition=[0, 8, 0],
                                    baseOrientation=pb.getQuaternionFromEuler((math.pi / 2, math.pi / 2, 0)),
                                    useFixedBase=1)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # xin = pb.addUserDebugParameter("x", -0.224, 0.224, 0)
        # yin = pb.addUserDebugParameter("y", -0.224, 0.224, 0)
        # zin = pb.addUserDebugParameter("z", 0, 1., 0.5)

    def spawn_agent(self):
        agent = Agent(urdf_path="urdfs/agent.urdf")
        self.env_pb_objects.append(agent.id)
        pass

    def run(self):

        # spawn obstacles randomly
        def spawn_random_obstacle():
            # spawns random obstacle from directory at the horizon
            pass

        enough_data_created = False
        while 100:  # not enough_data_created:
            pass

        print("Simulation stopped")
        return None


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

    test_show_obstacles()
