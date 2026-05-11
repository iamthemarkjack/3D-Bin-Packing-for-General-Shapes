import yaml
import numpy as np

from env import PyBulletPackingEnv
from evaluate import PlacementEvaluator
from opt import BoTorchOptimizer
from objects import load_objects


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["seed"])

objects = load_objects(cfg["paths"]["object_dir"])
env = PyBulletPackingEnv(cfg)
evaluator = PlacementEvaluator(cfg, env)
optimizer = BoTorchOptimizer(cfg)

margin = cfg["bin"]["margin"]
bw = cfg["bin"]["width"]
bd = cfg["bin"]["depth"]

bounds = np.array([
    [-bw / 2 + margin, bw / 2 - margin],
    [-bd / 2 + margin, bd / 2 - margin],
])

for obj_idx in range(cfg["experiment"]["num_objects"]):
    obj = objects[obj_idx % len(objects)]
    print()
    print("=" * 60)
    print(f"OBJECT {obj_idx + 1}")
    print("=" * 60)

    for it in range(cfg["bo"]["iterations"]):
        candidate = optimizer.suggest(bounds)
        metrics = evaluator.evaluate(candidate[0], candidate[1], obj)
        optimizer.observe(candidate, metrics)
        score = optimizer.objective(metrics)

        print(
            f"iter={it:02d} "
            f"x={candidate[0]:.3f} "
            f"y={candidate[1]:.3f} "
            f"score={score:.6f}"
        )

    best_idx = np.argmin(optimizer.Y)
    best_xy = optimizer.X[best_idx]
    env.place_object( best_xy[0], best_xy[1], obj, temporary=False)

    for _ in range(400):
        env.step()