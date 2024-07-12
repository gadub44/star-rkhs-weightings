from __future__ import annotations

import math
import numpy as np
import scipy as sp
import seaborn ; seaborn.set()
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings

from Model import RKHSWeighting
from Loss import *


class Learner:
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    loss : Loss
        Optimisation loss.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.
    """
    def __init__(self, loss: Loss, rng=None):
        self.loss = loss
        self.rng = np.random.default_rng(rng)
    
    def fit_model(self, X: np.ndarray, y: np.ndarray, model: RKHSWeighting, **kwargs) -> RKHSWeighting: 
        self.data_ = X
        self.targets_ = y
        self.model_ = model
        return self._fit_model(**kwargs)

    def _fit_model(self, **kwargs) -> RKHSWeighting:
        raise NotImplementedError
    
    def _get_n_samples(self) -> int:
        return self.data_.shape[0]
    
    def _get_n_dim(self) -> int:
        return self.data_.shape[1]
    
    def _get_lipschitz(self) -> float:
        return self.loss.lipschitz(self.model_.max_output())
    
    def _get_delta(self) -> float:
        return 0.05 

    def _sample_batch(self, batch_size):
        sample_size, n_features = self.data_.shape
        batch_idx = self.rng.choice(sample_size, size=batch_size, replace=True)
        batch_data = self.data_[batch_idx, :].reshape((batch_size, n_features))
        batch_targets = self.targets_[batch_idx]
        return batch_data, batch_targets 
    
    def rademacher_complexity(self) -> float:
        norm = self.model_.norm()
        theta = self.model_.operator_norm()
        m = self._get_n_samples()
        return norm *  theta / math.sqrt(m)

    def rademacher_bound(self) -> float:
        """
        Probabilistic bound on the generalization error.
        """   
        max_output = self.model_.max_output()
        m = self._get_n_samples()
        rho = self._get_lipschitz()
        delta = self._get_delta()
        max_loss = self.loss.max_value(max_output)
        min_loss = self.loss.min_value(max_output)
        term1 = self.training_loss()
        term2 = 2 * rho * self.rademacher_complexity()
        term3 = (max_loss-min_loss) * math.sqrt(math.log(1 / delta) / (2 * m))
        return term1 + term2 + term3

    def _model_training_loss(self, model: RKHSWeighting) -> np.ndarray:
        output = model.output(self.data_)
        targets = self.targets_
        return self.loss.calculate(output, targets)    
    
    def training_loss(self) -> np.ndarray:
        return self._model_training_loss(self.model_)
    
    def _set_model(self, new_model: RKHSWeighting):
        """
        Utility function that directly sets self.model to new_model.
        """
        self.model_ = new_model
    

class SFGDLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    loss : Loss
        Optimisation loss.

    n_iter : int, default=100
        Number of iterations of the stochastic functional gradient descent

    B : {'auto'} or int, default=1000
        Maximal RKHS norm of the weight function.
        If 'auto', then B = sqrt(m)/theta.

    regularization : {'auto'} or float, default='auto'
        Tikhonov regularization parameter.
        If 'auto', then it will be equal to the value suggested by the convergence bounds.

    stepsize : {'auto'} or float, default='auto'
        Stepsize of the gradient descent.
        If 'auto', then it will be equal to the value suggested by the convergence bounds.

    batch_size : int, default=32
        Number of examples sampled every iteration for approximating the functional gradient.

    apply_projection_step : boolean, default=True
        Whether to bound by B the iterates of the gradient descent.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.
    
    """
    def __init__(self, loss=LogisticLoss(), 
                 n_iter=100, B=1000, regularization='auto', 
                 stepsize='auto', batch_size=32,
                 apply_projection_step=True, 
                 rng=None, **kwargs):
        super().__init__(loss=loss, rng=rng)
        self.n_iter = n_iter
        self.B = B
        self.regularization = regularization
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.apply_projection_step = apply_projection_step
    
    def _fit_model(self) -> RKHSWeighting:
        self.iteration_model_ = self.model_.copy()
        self.n_total_iter_ = 0
        self.n_of_average_ = 1
        self.iterate_norms_ = [0.0]
        self.iterate(self.n_iter)
        return self.model_

    def iterate(self, n_iter):
        for _ in range(n_iter):
            self._take_one_step(self.batch_size)
    
    def _take_one_step(self, batch_size):
        center = self.iteration_model_.sample_center()
        batch_data, batch_targets = self._sample_batch(batch_size)
        batch_output = self.iteration_model_.output(batch_data)
        derivative = self.loss.derivative(batch_output, batch_targets)
        base_preds = self.iteration_model_.base_pred.eval([center], batch_data)
        coef = -self._get_stepsize() * derivative * base_preds.flatten()
        scale = 1.0 - self._get_stepsize() * self._get_regularization()
        self.iteration_model_.efficient_update(center, np.mean(coef), scale)
        if self.apply_projection_step:
            self.iteration_model_.project(self._get_B())
        self.model_.update_average(self.iteration_model_, self.n_of_average_)
        self.iterate_norms_.append(self.iteration_model_.norm())
        self.n_total_iter_ += 1
        self.n_of_average_ += 1 

    def _get_B(self) -> float:
        if self.B == 'auto':
            return math.sqrt(self._get_n_samples()) / self.model_.operator_norm()
        else:
            return float(self.B)
    
    def _get_stepsize(self) -> float:
        if self._use_slow_sgd() and self.stepsize == 'auto':
            return self._get_B() / (self._get_lipschitz() * self.model_.kappa() * math.sqrt(self.n_iter))
        elif self.stepsize == 'auto':
            reg = self._get_regularization()
            return 1 / (reg * (self.n_total_iter_ + 1))
        else:
            return float(self.stepsize)
    
    def _get_regularization(self) -> float:
        if self._use_slow_sgd():
            return 0
        elif self.regularization == 'auto':
            rho = self._get_lipschitz()
            B = self._get_B()
            m = self._get_n_samples()
            return math.sqrt(8) * rho / (B * math.sqrt(m))
        else:
            return float(self.regularization)

    def _get_lipschitz(self) -> float:
        # Importantly, it must be self.iteration_model below rather than self.model.
        # Otherwise, a huge slowdown will happen because the norm of self.model is
        # not efficiently updated every iteration.
        return self.loss.lipschitz(self.iteration_model_.max_output())

    def _get_max_iterate_norm(self) -> float:
        return max(self.iterate_norms_)

    def _use_slow_sgd(self) -> bool:
        return self.regularization == 0
    
    def stability_bound(self):
        """
        Bound in expectation on the convergence of the gradient descent.
        """
        reg = self._get_regularization()
        if not self._is_stability_bound_valid():
            return "Stability bound is invalid."
        B = self._get_max_iterate_norm()
        m = self._get_n_samples()
        T = self.n_total_iter_
        rho = self._get_lipschitz()
        theta = self.model_.operator_norm()
        kappa = self.model_.kappa()
        term1 = 2 * rho * theta * B / math.sqrt(m)
        term2 = reg * B**2
        term3 = 8 * rho**2 / (reg * m)
        term4 = (1 + math.log(T)) * (rho * kappa + reg * B)**2 / (2 * reg * T)
        bound = term1 + term2 + term3 + term4
        return bound

    def _is_stability_bound_valid(self):
        return self._get_regularization() > 0 and self.stepsize == 'auto'

    def slow_sgd_bound(self):
        if self.is_slow_sgd_bound_valid():
            return self._get_B() * self._get_lipschitz() * self.model_.kappa() / math.sqrt(self.n_iter)
        else:
            return 'Slow SGD bound does not apply.'
        
    def is_slow_sgd_bound_valid(self):
        return self._use_slow_sgd() and self.stepsize == 'auto'
    
    def _set_model(self, new_model: RKHSWeighting):
        """
        Utility function that directly sets self.model to new_model.
        """
        self.model_ = new_model
        self.iteration_model_ = new_model.copy()
        self.n_total_iter_ = 0
        self.n_of_average_ = 1
        self.iterate_norms_.append(new_model.norm())


class LeastSquaresLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.

    eps : float, default=1e-10
        Parameter to ensure numerical stability.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.
    
    """
    def __init__(self, n_iter=100, regularization=1e-5, eps=1e-10, rng=None, **kwargs):
        super().__init__(loss=MSE(), rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization
        self.eps = eps
    
    def _fit_model(self) -> RKHSWeighting: 
        """
        Learn the optimal coefficients.

        The regularization term is the RKHS norm of the weight function.
        """
        model = self.model_
        X = self.data_
        y = self.targets_
        T = self.n_iter
        centers = [model.sample_center() for _ in range(T)]
        model.set_centers(centers)
        Phi = model.expectations(X)
        I = np.eye(T)
        G = model.gram()
        m, _ = X.shape   
        A = np.dot(Phi.T, Phi) + m * self.regularization * G + m * self.eps * I
        B = np.dot(Phi.T, y)
        coefs = np.linalg.solve(A, B).tolist()
        model.set_coefs(coefs)
        return model


class LassoLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.    
    """
    def __init__(self, n_iter=100, regularization=1e-5, rng=None, **kwargs):
        super().__init__(loss=MSE(), rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_model(self) -> RKHSWeighting: 
        model = self.model_
        X = self.data_
        y = self.targets_
        centers = [model.sample_center() for _ in range(self.n_iter)]
        model.set_centers(centers)
        Phi = model.expectations(X) 
        lasso = Lasso(alpha=self.regularization, fit_intercept=False, max_iter=5*len(centers))
        lasso.fit(Phi, y)
        coefs = lasso.coef_.copy().tolist()
        model.set_coefs(coefs)
        model.remove_useless_centers()
        return model


class OptimalStepsizeLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.

    use_batch : boolean, default=True
        Whether to approximate the gradient on a batch, rather than the whole dataset

    batch_size : int, default=100
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.    
    """
    def __init__(self, loss=MSE(), n_iter=100, regularization=1e-5, 
                 use_batch=True, batch_size=100, rng=None, **kwargs):
        super().__init__(loss=loss, rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization
        self.use_batch = use_batch
        self.batch_size = batch_size

    def _fit_model(self) -> RKHSWeighting:
        self.iterate(self.n_iter)
        return self.model_

    def iterate(self, n_iter):
        for _ in range(n_iter):
            self._take_one_step()
    
    def _take_one_step(self):
        center = self.model_.sample_center()
        coef = self._get_coef(center)
        self.model_ = (1-self.regularization) * self.model_
        self.model_.add_center(center, coef)
    
    def _get_coef(self, center) -> float:
        center_model = self.model_.empty_copy().set_centers([center], [1])
        if self.use_batch:
            batch_data, batch_targets = self._sample_batch(self.batch_size)
        else:
            batch_data, batch_targets = self.data_, self.targets_

        model_outputs = self.model_.output(batch_data)
        center_model_outputs = center_model.output(batch_data)
        weight_func_at_w = self.model_.eval_weight_func(center)
        reg = self.regularization
        Kww = center_model.norm()**2
        m, _ = batch_data.shape

        if isinstance(self.loss, MSE):
            num = 1/m * np.dot((1-reg) * model_outputs - batch_targets, center_model_outputs) + reg * (1-reg) * weight_func_at_w
            denom = 1/m * np.sum(center_model_outputs**2) + reg * Kww
            return -num / denom
        else:
            loss = self.loss

            def derivative_function(eta):
                value = reg * (1-reg) * (weight_func_at_w + eta * Kww)
                outputs = (1-reg)*model_outputs + eta * center_model_outputs
                partial_derivatives = loss.derivative(output=outputs, targets=batch_targets)
                value += 1/m * np.dot(partial_derivatives, center_model_outputs)
                return value

            return sp.optimize.root_scalar(derivative_function, x0=0, x1=50, maxiter=None).root