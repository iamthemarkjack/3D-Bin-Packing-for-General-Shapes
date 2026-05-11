import os
import glob
import numpy as np

from utils import ObjectSpec


def bbox_from_obj_file(obj_path):
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return (0.05, np.array([0.1, 0.1, 0.1]), np.zeros(3))

    v = np.array(verts)
    mn = v.min(0)
    mx = v.max(0)
    extents = mx - mn
    
    return (extents[2] / 2.0, extents, mn)


def load_objects(obj_dir):
    obj_files = sorted(
        glob.glob(os.path.join(obj_dir, "*.obj"))
    )

    objects = []
    for obj_path in obj_files:
        name = os.path.splitext(os.path.basename(obj_path))[0]
        half_h, extents, min_xyz = bbox_from_obj_file(obj_path)
        objects.append(ObjectSpec(name=name, obj_path=obj_path, 
                                  half_h=half_h, extents=extents, min_xyz=min_xyz))

    return objects