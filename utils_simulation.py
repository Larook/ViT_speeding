
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
        agent.get_image_from_camera()
        pb.stepSimulation()
        time.sleep(0.005)
