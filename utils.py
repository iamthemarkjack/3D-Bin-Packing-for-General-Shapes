from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectSpec:
    name: str
    obj_path: str
    half_h: float
    extents: np.ndarray
    min_xyz: np.ndarray


@dataclass
class PlacementMetrics:
    max_height: float
    displacement: float
    speed_penalty: float