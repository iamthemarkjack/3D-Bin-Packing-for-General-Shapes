import time
import numpy as np
import pybullet as p

from utils import PlacementMetrics


class PlacementEvaluator:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env


    def evaluate(self, x, y, obj):
        state_id = self.env.save_state()
        trial = self.env.place_object(x, y, obj, temporary=True)
        spawn_xy = np.array([x, y])
        speed_integral = 0.0

        for step in range(self.cfg["physics"]["settle_steps"]):
            self.env.step()
            if step >= self.cfg["physics"]["settle_steps"] - self.cfg["physics"]["measure_steps"]:
                lv, _ = p.getBaseVelocity(
                    trial,
                    physicsClientId=self.env.client,
                )
                speed_integral += np.linalg.norm(lv)

        final_pos, _ = p.getBasePositionAndOrientation(trial, physicsClientId=self.env.client)

        metrics = PlacementMetrics(
            max_height=max(
                self.env.get_max_height(),
                final_pos[2] + obj.half_h,
            ),
            displacement=np.linalg.norm(
                np.array(final_pos[:2]) - spawn_xy
            ),
            speed_penalty=(0.1 * speed_integral / self.cfg["physics"]["measure_steps"]
            ),
        )

        p.removeBody(trial, physicsClientId=self.env.client)
        self.env.restore_state(state_id)

        return metrics