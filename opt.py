import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class BoTorchOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.dtype = torch.double
        self.X = []
        self.Y = []


    def objective(self, metrics):
        return (
            self.cfg["objective"]["height_weight"] * metrics.max_height
            + self.cfg["objective"]["instability_weight"] * (metrics.displacement + metrics.speed_penalty)
        )


    def observe(self, x, metrics):
        score = self.objective(metrics)
        self.X.append(x)
        self.Y.append(score)
        

    def fit_model(self):
        train_X = torch.tensor(np.array(self.X), dtype=self.dtype, device=self.device)
        y = np.array(self.Y)
        y = (y - y.mean()) / (y.std() + 1e-8)

        train_Y = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=2),
            outcome_transform=Standardize(m=1),
        )

        model.likelihood.noise = float(self.cfg["bo"]["noise"])
        model.likelihood.raw_noise.requires_grad_(False)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model, train_X
    

    def suggest(self, bounds):
        if len(self.X) < self.cfg["bo"]["init_points"]:
            return np.random.uniform(bounds[:, 0], bounds[:, 1])

        model, train_X = self.fit_model()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        acq = qLogNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)
        bounds_t = torch.tensor(bounds.T, dtype=self.dtype, device=self.device)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_t,
            q=1,
            num_restarts=self.cfg["bo"]["restarts"],
            raw_samples=self.cfg["bo"]["raw_samples"],
        )

        return candidate.squeeze(0).cpu().numpy()