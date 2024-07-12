import math
import numpy as np
from numpy.random import default_rng
from scipy.stats import cauchy
from scipy.stats import multivariate_normal
from scipy.stats import norm


class Distribution:
    def __init__(self, n_dim, rng=None) -> None:
        self.n_dim = n_dim
        self.rng = default_rng(rng)

    def sample(self, n):
        raise NotImplementedError

    def score_samples(self, x):
        raise NotImplementedError


class GaussianDistribution(Distribution):
    def __init__(self, n_dim, mean=None, sigma=1.0, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.sigma = sigma
        if mean is None:
            self.mean = np.array([0]*self.n_dim)
        else:
            self.mean = mean

    def sample(self):
        return self.rng.normal(self.mean, self.sigma)

    def score_samples(self, x):
        rv = multivariate_normal(mean=self.mean, cov=self.sigma**2)
        return rv.pdf(x)

    def set_mean(self, mean):
        self.mean = mean


class GaussianThresholdDistribution(Distribution):
    def __init__(self, n_dim, sigma=1.0, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.sigma = sigma

    def sample(self):
        idx = self.rng.integers(self.n_dim)
        param = self.rng.normal(0, self.sigma)
        return (idx, param)
    
    def score_samples(self, x):
        rv = norm(loc=0, scale=self.sigma)
        return [rv.pdf(z) / self.n_dim for (_, z) in x]


class UniformThresholdDistribution(Distribution):
    def __init__(self, n_dim, min_val, max_val, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.min_val = min_val
        self.max_val = max_val

    def sample(self):
        idx = self.rng.integers(self.n_dim)
        param = self.rng.uniform(self.min_val, self.max_val)
        return (idx, param)
    
    def score_samples(self, x):
        return len(x) * [1 / (self.n_dim * (self.max_val - self.min_val))]


class CauchyDistribution(Distribution):
    def __init__(self, n_dim, rng=None) -> None:
        super().__init__(n_dim, rng)

    def sample(self):
        return self.rng.standard_cauchy(size=self.n_dim)

    def score_samples(self, x):
        rv = cauchy()
        return rv.pdf(x)


    