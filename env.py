import pybullet as p
import pybullet_data

from heightmap import HeightMap


class PyBulletPackingEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = p.connect(p.GUI if cfg["physics"]["gui"] else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.resetSimulation(physicsClientId=self.client)

        p.setGravity(0, 0, cfg["physics"]["gravity"], physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)

        self.heightmap = HeightMap(cfg, self.client)
        self.bodies = []
        self.build_bin()

        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0.15],
            physicsClientId=self.client,
        )


    def build_bin(self):
        bw = self.cfg["bin"]["width"]
        bd = self.cfg["bin"]["depth"]
        bh = self.cfg["bin"]["height"]
        wt = self.cfg["bin"]["wall_thickness"]

        floor_rgba = [0.75, 0.75, 0.75, 1.0]
        wall_rgba = [0.65, 0.65, 0.65, 0.12]

        def box(half_extents, pos, rgba):
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client,
            )

            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=rgba,
                physicsClientId=self.client,
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.client,
            )

        wall_hw = bh / 2
        wall_z = wall_hw

        box([bw / 2 + wt, bd / 2 + wt, wt / 2], [0, 0, -wt / 2], floor_rgba)
        box([wt / 2, bd / 2 + wt, wall_hw], [-bw / 2 - wt / 2, 0, wall_z], wall_rgba)
        box([wt / 2, bd / 2 + wt, wall_hw], [bw / 2 + wt / 2, 0, wall_z], wall_rgba)
        box([bw / 2, wt / 2, wall_hw], [0, -bd / 2 - wt / 2, wall_z], wall_rgba)
        box([bw / 2, wt / 2, wall_hw], [0, bd / 2 + wt / 2, wall_z], wall_rgba)


    def save_state(self):
        return p.saveState(physicsClientId=self.client)
    

    def restore_state(self, state_id):
        p.restoreState(stateId=state_id, physicsClientId=self.client)
        p.removeState(state_id, physicsClientId=self.client)


    def place_object(self, x, y, obj, temporary=False):
        surface_h = self.heightmap.query(x, y)
        z0 = surface_h - obj.min_xyz[2] + self.cfg["placement"]["z_offset"]

        col = p.createCollisionShape(p.GEOM_MESH, fileName=obj.obj_path, physicsClientId=self.client)

        rgba = (
            [0.9, 0.2, 0.2, 0.45]
            if temporary
            else [0.2, 0.65, 0.8, 1.0]
        )

        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj.obj_path, rgbaColor=rgba, physicsClientId=self.client)

        body = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z0],
            physicsClientId=self.client,
        )

        p.changeDynamics(
            body,
            -1,
            lateralFriction=self.cfg["placement"]["friction"],
            rollingFriction=self.cfg["placement"]["rolling_friction"],
            restitution=self.cfg["placement"]["restitution"],
            physicsClientId=self.client,
        )

        if not temporary:
            self.bodies.append(body)

        return body
    

    def step(self):
        p.stepSimulation(physicsClientId=self.client)


    def get_max_height(self):
        if not self.bodies:
            return 0.0
        tops = []
        for body in self.bodies:
            aabb = p.getAABB(body, physicsClientId=self.client)
            tops.append(aabb[1][2])
            
        return max(tops)