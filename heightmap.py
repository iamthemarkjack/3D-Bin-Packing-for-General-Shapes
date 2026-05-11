import numpy as np
import pybullet as p


class HeightMap:
    def __init__(self, cfg, client):
        self.client = client
        self.res = cfg["heightmap"]["resolution"]
        self.near = cfg["heightmap"]["near"]
        self.bin_h = cfg["bin"]["height"]
        self.far = self.bin_h + cfg["heightmap"]["far_offset"]
        self.bin_cx = 0.0
        self.bin_cy = 0.0
        self.cam_z = self.far + 0.01
        self.view = p.computeViewMatrix(
            cameraEyePosition=[
                self.bin_cx,
                self.bin_cy,
                self.cam_z,
            ],
            cameraTargetPosition=[
                self.bin_cx,
                self.bin_cy,
                0.0,
            ],
            cameraUpVector=[1, 0, 0],
        )
        self.proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, 
                                                 nearVal=self.near, farVal=self.far)
        fov_rad = np.deg2rad(60)
        self.half_span = self.cam_z * np.tan(fov_rad / 2)


    def depth_to_z(self, depth_buf):
        z_cam = (self.near * self.far / (self.far
                - depth_buf * (self.far - self.near)
            ))
        
        return self.cam_z - z_cam
    

    def query(self, x0, y0):
        _, _, _, depth, _ = p.getCameraImage(
            self.res,
            self.res,
            viewMatrix=self.view,
            projectionMatrix=self.proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )

        depth = np.array(depth).reshape(self.res, self.res)
        z_world = self.depth_to_z(depth)
        xs = np.linspace(self.bin_cx + self.half_span, self.bin_cx - self.half_span, self.res)
        ys = np.linspace(self.bin_cy - self.half_span, self.bin_cy + self.half_span, self.res)

        x_idx = np.interp(x0, xs[::-1], np.arange(self.res)[::-1])
        y_idx = np.interp(y0, ys, np.arange(self.res))

        x0i = int(np.clip(x_idx, 0, self.res - 2))
        y0i = int(np.clip(y_idx, 0, self.res - 2))

        dx = x_idx - x0i
        dy = y_idx - y0i

        h = (
            z_world[x0i, y0i] * (1 - dx) * (1 - dy)
            + z_world[x0i + 1, y0i] * dx * (1 - dy)
            + z_world[x0i, y0i + 1] * (1 - dx) * dy
            + z_world[x0i + 1, y0i + 1] * dx * dy
        )

        return float(h)