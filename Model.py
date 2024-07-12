from __future__ import annotations

from copy import deepcopy
import math
import matplotlib.pyplot as plot
import numpy as np
import scipy as sp
from scipy.special import erf, binom
from sklearn.svm import LinearSVC
from typing import List

from utils import *
from BasePredictor import *
from Distribution import *
from Kernel import *
from Loss import *
from WeightFunction import WeightFunction, Center

import visuals


class RKHSWeighting(WeightFunction):
    """
    RKHS weighting of functions model. 
    
    This is the generic class, upon which are built specific instantiations of the model.

    See the original paper : https://hal.science/hal-04236058v1

    Parameters
    ----------
    dist : Distribution
        Probability distribution for sampling the random parameters. "p" in the paper.

    kernel : Kernel
        Similarity kernel defining the RKHS. "K" in the paper.

    base_pred : BasePredictor
        The functions in "RKHS weightings of functions". "phi" in the paper.

    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    max_n_mc : int, default=10000
        Maximal number of samples for Monte Carlo estimations.

    n_mc : int or None, default=None
        Number of samples for Monte Carlo estimations.
        If None, a reasonable number will be calculated automatically according to the parameters of the instantiations,
        up to max_n_mc.

    use_mc : boolean, default=True
        Whether to use Monte Carlo estimations.
        Must only be set to False when a "_exact_expectations" function is defined in the inheriting class.

    mc_precision : float, default=0.01
        Desired absolute precision for Monte Carlo estimations

    Inherits:
    ----------
    WeightFunction: Refer to the docstring of WeightFunction for details on its parameters and attributes.

    """
    def __init__(self, dist: Distribution, kernel: Kernel, base_pred: BasePredictor, rng=None,
                 max_n_mc=10000, n_mc=None, use_mc=True, mc_precision=0.01) -> None:
        self.dist = dist
        self.base_pred = base_pred
        self.rng = np.random.default_rng(rng)
        self.max_n_mc = max_n_mc
        self.n_mc = n_mc
        self.use_mc = use_mc
        self.mc_precision = mc_precision
        super().__init__(kernel=kernel)
    
    def expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        """
        Returns the (n_examples, n_centers) array containing
        the value of the expectation for each (center, x) pair.
        """
        if self.use_mc:
            expectations = self._mc_expectations(ensure_X_2d(X), self.n_mc, centers)
        else:
            expectations = self._exact_expectations(ensure_X_2d(X), centers)
        return expectations

    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        """
        This function should ideally be implemented
        for any new instantiation. Otherwise, Monte Carlo
        will be used, which is slower and less accurate.

        Use self.set_use_mc(False) to use this function.
        """
        raise NotImplementedError
    
    def _mc_expectations(self, X: np.ndarray, n_mc=None, centers: List[Center] = None, batch_size=1000) -> np.ndarray:
        """
        Calculate the expectations using Monte Carlo
        """
        if n_mc is None:
            n_mc = self.get_expect_n_mc()

        if n_mc <= batch_size:
            # If n_mc is small enough calculate in a single batch
            return (1 / n_mc) * self._partial_mc_expectations(X, n_mc)
        else:
            # Calculate in batches
            output = np.zeros(shape=(X.shape[0], self.get_n_centers()))
            num_batches = n_mc // batch_size
            remainder = n_mc % batch_size

            for _ in range(num_batches):
                output += (1 / n_mc) * self._partial_mc_expectations(X, batch_size, centers)

            if remainder > 0:
                # Calculate the remaining samples
                output += (1 / n_mc) * self._partial_mc_expectations(X, remainder, centers)

            return output
    
    def _partial_mc_expectations(self, X: np.ndarray, n_mc, centers: List[Center] = None) -> np.ndarray:
        list_of_w = [self.sample_center() for _ in range(n_mc)]
        base_predictions = self.base_pred.eval(list_of_w, X) # m x N
        gram = self.kernel.calculate(list_of_w, self.get_center_params(centers))
        return np.dot(base_predictions, gram)

    def get_expect_n_mc(self):
        if self.n_mc is None:
            target_var = self.mc_precision**2
            max_center_norm = self.get_max_center_norm()
            n_mc = int(min(max_center_norm**2 * self.kappa()**2 / target_var, self.max_n_mc))
        else:
            n_mc = self.n_mc
        return n_mc

    def get_empty_expectations(self, X, centers: List[Center] = None):
        m = X.shape[0]
        T = self.get_n_centers() if centers is None else len(centers)
        return np.zeros(shape=(m, T))
    
    def output(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        """
        Returns the size m array containing
        the output of the prediction model for all examples in X.

        Equations-wise, this is Lambda alpha(X).
        """
        X = ensure_X_2d(X)
        if self.get_n_centers() == 0:
            output = np.zeros(shape=X.shape[0])
        else:
            if self.use_mc:
                output = self.mc_output(X)
            else:
                coefs = self.get_coefs(centers)
                output = np.dot(self.expectations(X, centers), coefs).flatten()
        return output
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.output(X)

    def mc_output(self, X: np.ndarray, n_mc=None, batch_size=1000, centers: List[Center] = None) -> np.ndarray:
        if centers is not None:
            print('Monte Carlo output not compatible with custom centers')
        """
        Calculates model output using Monte Carlo
        """
        X = ensure_X_2d(X) 
        if n_mc is None:
            n_mc = self.get_output_n_mc()
        if n_mc <= batch_size:
            # If n_mc is small enough calculate in a single batch
            return (1 / n_mc) * self._partial_mc_output(X, n_mc)
        else:
            # Calculate in batches
            output = np.zeros(X.shape[0])
            num_batches = n_mc // batch_size
            remainder = n_mc % batch_size

            for _ in range(num_batches):
                output += (1 / n_mc) * self._partial_mc_output(X, batch_size)

            if remainder > 0:
                # Calculate the remaining samples
                output += (1 / n_mc) * self._partial_mc_output(X, remainder)

            return output

    def _partial_mc_output(self, X: np.ndarray, n_mc) -> np.ndarray:
        list_of_w = [self.sample_center() for _ in range(n_mc)]
        base_predictions = self.base_pred.eval(list_of_w, X)
        weight_func_values = self.eval_weight_func_multiple_centers(list_of_w)
        return np.dot(base_predictions, weight_func_values)

    def get_output_n_mc(self):
        if self.n_mc is None:
            target_var = self.mc_precision**2
            n_mc = int(self.norm()**2 * self.kappa()**2 / target_var)
            n_mc = int(min(n_mc, self.max_n_mc))
        else:
            n_mc = self.n_mc
        return n_mc

    def efficient_update(self, center, coef, scale=1.0):
        """
        Multiplies the weight function by scale, then add
        the new center and coefficient.

        The scale is useful when regularizing.
        """
        self *= scale
        return self.add_center(center, coef)

    def theta(self):
        try:
            return self.theta_ 
        except:
            try:
                self.theta_ = min(self.true_theta(), self.iota())
            except NotImplementedError:
                self.theta_ = self.iota()
        return self.theta_
    
    def true_theta(self):
        """
        Calculates theta using an analytical formula.
        Each instantation may implement this function.
        Otherwise, constant will be calculated using Monte Carlo.

        """
        raise NotImplementedError
    
    def mc_theta(self, X: np.ndarray, n_mc=None) -> float:
        """
        Estimate constant theta using Monte Carlo.
        """
        m = X.shape[0]
        if type(n_mc) is int:
            T = n_mc
        else:
            T = self.max_n_mc

        # Sample centers for outer and inner expectation
        centers_outer = [self.sample_center() for _ in range(T)]
        centers_inner = [self.sample_center() for _ in range(T)]

        # Calculate kernel diagonal and find nonzero indices
        kern_diag = self.kernel.diag(centers_outer, centers_inner)
        nonzero_idx = np.flatnonzero(kern_diag)

        if len(nonzero_idx) == 0:
            return 0
        
        # Initialize arrays for base predictions phi(w, x)
        base_pred_outer = np.zeros((m, T))
        base_pred_inner = np.zeros((m, T))

        # Compute base predictions for nonzero indices
        base_pred_outer[:, nonzero_idx] = self.base_pred.eval([centers_outer[i] for i in nonzero_idx], X)
        base_pred_inner[:, nonzero_idx] = self.base_pred.eval([centers_inner[i] for i in nonzero_idx], X)
        
        # Compute one candidate theta (squared) for each example
        candidates_squared = np.mean(base_pred_outer * base_pred_inner * kern_diag, axis=1)

        # theta is the supremum over X, therefore return the maximal candidate
        return float(np.sqrt(max(candidates_squared)))

    def iota(self):
        try:
            return self.iota_
        except:
            try:
                self.iota_ = self.true_iota()
            except NotImplementedError:
                self.iota_ = self.kappa()
        return self.iota_
    
    def true_iota(self):
        """
        Calculates iota using an analytical formula.
        Each instantation may implement this function.
        Otherwise, constant will be calculated using Monte Carlo.

        """
        raise NotImplementedError
    
    def mc_iota(self, X: np.ndarray, n_mc=None) -> float:
        """
        Estimate constant iota using Monte Carlo.
        """
        m = X.shape[0]
        if type(n_mc) is int:
            T = n_mc
        else:
            T = self.max_n_mc
        centers = [self.sample_center() for _ in range(T)]
        kern_diag = self.kernel.diag(centers, centers)
        nonzero_idx = np.flatnonzero(kern_diag)
        if len(nonzero_idx) == 0:
            return 0
        base_predictions = np.zeros((m, T))
        nonzero_centers = [centers[i] for i in nonzero_idx]
        base_predictions[:, nonzero_idx] = self.base_pred.eval(nonzero_centers, X) # m x T
        rdv = np.sqrt(base_predictions**2 * kern_diag)
        candidates = np.mean(rdv, axis=1)
        return float(max(candidates))

    def kappa(self):
        try:
            return self.kappa_
        except:
            try:
                self.kappa_ = self.true_kappa()
                return self.kappa_
            except NotImplementedError:
                print("kappa could not be calculated")
                return math.inf
    
    def true_kappa(self):
        """
        Calculates kappa using an analytical formula.
        Each instantation may implement this function.
        Otherwise, constant will be calculated using Monte Carlo.

        """
        raise NotImplementedError
    
    def mc_kappa(self, X: np.ndarray, n_mc=None) -> float:
        """
        Estimate constant kappa using Monte Carlo.
        """
        m = X.shape[0]
        if type(n_mc) is int:
            T = n_mc
        else:
            T = self.max_n_mc
        centers = [self.sample_center() for _ in range(self.max_n_mc)]
        kern_diag = self.kernel.diag(centers, centers)
        nonzero_idx = np.flatnonzero(kern_diag)
        if len(nonzero_idx) == 0:
            return 0
        base_predictions = np.zeros((m, T))
        nonzero_centers = [centers[i] for i in nonzero_idx]
        base_predictions[:, nonzero_idx] = self.base_pred.eval(nonzero_centers, X) # m x T
        rdv = base_predictions**2 * kern_diag
        candidates_squared = np.mean(rdv, axis=1)
        return float(np.sqrt(max(candidates_squared)))
    
    def operator_norm(self):
        return min(self.theta(), self.iota(), self.kappa())

    def get_n_dim(self):
        return self.dist.n_dim

    def sample_center(self):
        return self.dist.sample()

    def max_output(self):
        return self.norm() * self.operator_norm()

    def set_use_mc(self, boolean: bool):
        self.use_mc = boolean

    def set_mc_precision(self, precision: float):
        self.mc_precision = precision

    def set_max_n_mc(self, max_n: int):
        self.max_n_mc = max_n

    def set_n_mc(self, n_mc: int):
        self.n_mc = n_mc

    def copy(self) -> RKHSWeighting:
        return deepcopy(self)

    def learn_psi_of_x(self, x, T=1000):
        """
        Learns alpha \approx psi(x)
        """
        self.list_of_params = [self.sample_center() for _ in range(T)]
        Psi = self.expectations(x)
        G = self.gram()
        self.coefs = np.linalg.solve(G, Psi.T).flatten()
        self.norm_is_up_to_date = False

    def norm_of_psi_of_x(self, x, n_mc=10000):
        max_n_mc = 5000
        n_cycles = int(n_mc // max_n_mc)
        remainder = int(n_mc % max_n_mc)
        norm_squared = 0
        for _ in range(n_cycles):
            norm_squared += self._partial_norm_squared_of_psi_of_x(x, max_n_mc)
        norm_squared += self._partial_norm_squared_of_psi_of_x(x, remainder)
        norm_squared /= n_mc
        if norm_squared < 0:
            norm_squared = 0
        return float(np.sqrt(norm_squared))
    
    def _partial_norm_squared_of_psi_of_x(self, x, n_mc):
        if n_mc > 0:
            centers_left = [self.sample_center() for _ in range(n_mc)]
            centers_right = [self.sample_center() for _ in range(n_mc)]
            base_pred_left = self.base_pred.eval(centers_left, x)
            base_pred_right = self.base_pred.eval(centers_right, x)
            zipped_centers = zip(centers_left, centers_right)
            line_centers = [(np.array(w).reshape(1, -1), np.array(u).reshape(1, -1)) for (w, u) in zipped_centers]
            kern_diag = np.array([float(self.kernel.calculate(w, u)) for (w, u) in line_centers])
            return np.sum(kern_diag * base_pred_left * base_pred_right)
        else:
            return 0
    
    def empty_copy(self):
        """
        Makes and returns a copy of the Model with empty self.list_of_params and self.coefs

        Much more efficient than using self.copy
        """
        copy = self.__class__(input=self.input, rng=self.rng)
        for key, value in vars(self).items():
            if key in ['coefs', 'list_of_params']:
                setattr(copy, key, [])
            else:
                setattr(copy, key, value)
        return copy



class RWSign(RKHSWeighting):
    """
    Instantiation of the model using Gaussian kernel and distribution, 
    and sign as the base predictor.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    gamma : {'auto'} or float, default='auto'
        Parameter of the Gaussian kernel.

    max_theta : float, default=0.5
        Parameter defining the relationship between sigma and gamma, when
        sigma and/or gamma is 'auto'.

    dist_mean : None or np.array, default=None
        Mean vector of the distribution p.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the Model __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    gamma : float
        The adjusted gamma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input, sigma='auto', gamma='auto', max_theta=0.5,
                 dist_mean=None, rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        self.max_theta = max_theta
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = GaussianDistribution(n_dim=self.n_dim, mean=dist_mean, sigma=sigma, rng=rng)
        kernel = GaussianKernel(gamma=gamma)
        base = Sign()
        super().__init__(dist, kernel, base, rng, use_mc=use_mc, **kwargs)
    
    def get_and_set_mean_parameter(self, X, y):
        mean = self._get_good_mean_parameter(X, y)
        self._set_mean_parameter(mean)

    def _get_good_mean_parameter(self, X, y):
        clf = self._fit_svm(X, y)
        return clf.coef_.flatten()

    def _set_mean_parameter(self, param):
        self.dist.set_mean(param)
    
    def _fit_svm(self, X, y):
        self.svm = LinearSVC(random_state=0).fit(X, y)
        return self.svm

    def _get_fitted_svm(self):
        return self.svm
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        n = self.n_dim
        w_0 = self.dist.mean
        s = self.sigma
        g = self.gamma


        s2g2 = s**2 + g**2
        sqrt2zeta = 1 / math.sqrt(0.5 * (1/s**2 + 1/g**2))
        global_coef = (1 + s**2 / g**2)**(-n/2) 

        W = self.get_center_params(centers)
        W_norms = self.get_param_norms(centers)

        W_prime = (s**2 * W + g**2 * w_0) / s2g2

        w_0_norm = np.sqrt(np.dot(w_0, w_0))
        if w_0_norm != 0:
            mixed_norms = np.linalg.norm(W / g**2 + w_0 / s**2, axis=1)
        else:
            mixed_norms = W_norms / g**2

        exp_norms = np.exp(-W_norms**2 / (2*g**2) \
                           -w_0_norm**2 / (2*s**2) \
                           +(s**2 * g**2 / (2*s2g2)) * mixed_norms**2)  
        arr = get_scalar_over_norm(X, W_prime)
        expects = global_coef * array_times_vector(erf(arr / sqrt2zeta), exp_norms, axis=1)
        
        return expects
    
    def _exact_centered_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        n = self.n_dim
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2

        W = self.get_center_params(centers)
        W_norms = self.get_param_norms(centers)
        exp_norms = np.exp(-W_norms**2 / (2*s2g2))

        arr = get_scalar_over_norm(X, W)
        erf_coef = 1 / (g * math.sqrt(2 * (1 + g**2 / s**2)))
        global_coef = (1 + s**2 / g**2)**(-n/2)  

        return global_coef * array_times_vector(erf(erf_coef * arr), exp_norms, axis=1)

    def true_theta(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 + 2*s**2 / g**2)**(-n/4)

    def true_iota(self):
        return 1.0

    def true_kappa(self):
        return 1.0

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)
            
    def get_adjusted_gamma(self, gamma):
        if gamma == 'auto':
            return self.get_gamma_from_theta(self.max_theta)
        else:
            return float(gamma)

    def get_gamma_from_theta(self, theta):
        n = self.n_dim
        return math.sqrt(2) * self.sigma / math.sqrt(theta**(-4/n) - 1)

    def get_gamma_from_expect(self, var):
        n = self.n_dim
        return self.sigma / math.sqrt(var**(-2/n) - 1)


class RWRelu(RKHSWeighting):
    """
    Instantiation of the model using Gaussian kernel and distribution, 
    and ReLU as the base predictor.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    gamma : {'auto'} or float, default='auto'
        Parameter of the Gaussian kernel.

    max_theta : float, default=0.5
        Parameter defining the relationship between sigma and gamma, when
        sigma and/or gamma is 'auto'.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the RKHSWeighting __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    gamma : float
        The adjusted gamma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input: np.ndarray, sigma='auto', gamma='auto', 
                 max_theta=0.5, rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        self.max_theta = max_theta
        self.max_data_norm = max(np.linalg.norm(input, axis=1))
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = GaussianDistribution(n_dim=self.n_dim, sigma=sigma, rng=rng)
        kernel = GaussianKernel(gamma=gamma)
        base = Relu()
        super().__init__(dist, kernel, base, rng, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        n = self.n_dim
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2

        W = self.get_center_params(centers)
        W_norms = self.get_param_norms(centers)
        X_norms = np.linalg.norm(X, axis=1)

        sqrtpi = math.sqrt(math.pi)
        zeta = math.sqrt(1 / (1/s**2 + 1/g**2))
        sqrt2zeta = math.sqrt(2) * zeta
        arr = (s**2 / s2g2) * get_scalar_over_norm(X, W)

        global_coef = (g**2 / s2g2)**(n/2) / (2 * sqrtpi)
        exp_norms = np.exp(-W_norms**2 / (2 * s2g2))
        big_parenthesis_term1 = sqrt2zeta * np.exp(-arr**2 / sqrt2zeta**2)
        big_parenthesis_term2 = sqrtpi * arr * (1 + erf(arr / sqrt2zeta))
        big_parenthesis = big_parenthesis_term1 + big_parenthesis_term2

        expectation = global_coef * array_times_vector(array_times_vector(big_parenthesis, exp_norms, axis=1), X_norms, axis=0)

        return expectation
    
    def true_theta(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 + 2*s**2 / g**2)**(-(n-1)/4) * s * self.max_data_norm / math.sqrt(2 * math.pi)

    def true_iota(self):
        return self.sigma * self.max_data_norm / math.sqrt(2 * math.pi)

    def true_kappa(self):
        return self.sigma * self.max_data_norm / math.sqrt(2)
    
    def get_adjusted_sigma(self, sigma):
        if sigma == 'auto':
            return 2 * math.sqrt(self.max_theta)
        else:
            return float(sigma)
    
    def get_adjusted_gamma(self, gamma):
        if gamma == 'auto':
            return self.get_gamma_from_theta(self.max_theta)
        else:
            return float(gamma)
    
    def get_gamma_from_theta(self, theta):
        n = self.n_dim
        return math.sqrt(2) * self.sigma / math.sqrt(theta**(-4/n) - 1)

    def get_gamma_from_expect(self, var):
        n = self.n_dim
        return self.sigma / math.sqrt(var**(-2/n) - 1)


class RWStumps(RKHSWeighting):
    """
    Instantiation of the model using stumps as the base predictor.
    
    Distribution is Uniform on the chosen variable, and Gaussian on the value of the threshold.

    Kernel is Gaussian on the threshold, 0 when the variables are different.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    gamma : {'auto'} or float, default='auto'
        Parameter of the Gaussian kernel.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the Model __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    gamma : float
        The adjusted gamma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input, sigma='auto', gamma='auto', rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = GaussianThresholdDistribution(n_dim=self.n_dim, sigma=sigma, rng=rng)
        kernel = IndicatorGaussianKernel(gamma=gamma)
        base = Stump()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        n = self.n_dim
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2

        W = self.get_center_params(centers)
        if W.shape[0] == 0:
            print(W)
            print(centers)
        W_indices = W[:, 0].astype(int)
        W_values = W[:, 1]

        zeta = math.sqrt(1 / (1/s**2 + 1/g**2))
        sqrt2zeta = math.sqrt(2) * zeta
        W2prime = s**2 / s2g2 * W_values

        coef = zeta / (s * n)
        exp_norms = np.exp(-W_values**2 / (2 * s2g2))
        erf_stump = erf((X[:,W_indices]-W2prime)/sqrt2zeta)

        return coef * array_times_vector(erf_stump, exp_norms, axis=1)

    def true_theta(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 + 2*s**2 / g**2)**(-1/4) / math.sqrt(n)

    def true_iota(self):
        return 1.0

    def true_kappa(self):
        return 1.0

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)
            
    def get_adjusted_gamma(self, gamma):
        return self.sigma if gamma == 'auto' else float(gamma)

    def plot_variable(self, idx=0, min=-3, max=3, n_points=1000, show=True):
        """
        Plots the contribution of variable 'idx' to the output of the model
        """
        valid_centers = []
        valid_coefs = []
        for i in range(self.get_n_centers()):
             if int(self.list_of_params[i][0]) == idx:
                 valid_centers.append(self.list_of_params[i])
                 valid_coefs.append(self.coefs[i])
        values = np.linspace(start=min, stop=max, num=n_points)
        points = np.zeros(shape=(n_points, self.n_dim))
        points[:, idx] = values

        new_model = self.copy()
        new_model.set_centers(valid_centers, valid_coefs)

        output = new_model.output(points)
        plot.plot(values, output)
        if show:
            plot.show()
        

class RWUniformStumps(RKHSWeighting):
    """
    Instantiation of the model using stumps as the base predictor.
    
    Distribution is Uniform on the chosen variable, and Uniform on the value of the threshold.

    Kernel is Gaussian on the threshold, 0 when the variables are different.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    gamma : {'auto'} or float, default='auto'
        Parameter of the Gaussian kernel.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the Model __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    gamma : float
        The adjusted gamma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input: np.ndarray, gamma='auto', rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = input.shape[1]
        min_val = self.min_val = np.min(input)
        max_val = self.max_val = np.max(input)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = UniformThresholdDistribution(n_dim=self.n_dim, min_val=min_val, max_val=max_val, rng=rng)
        kernel = IndicatorGaussianKernel(gamma=gamma)
        base = Stump()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        W = self.get_center_params(centers).reshape(-1, 2)
        W_idx = W[:, 0].astype(int)
        W_values = W[:, 1]
        X_values = X[:, W_idx]
        n = self.n_dim
        lower = self.min_val
        upper = self.max_val
        gamma = self.gamma

        X_values_normalized = (X_values - W_values) / (math.sqrt(2) * gamma)
        lower_bounds_normalized = (lower - W_values) / (math.sqrt(2) * gamma)
        upper_bounds_normalized = (upper - W_values) / (math.sqrt(2) * gamma)

        factor1 = math.sqrt(math.pi) / (2 * n * (upper - lower))
        factor2 = 2 * erf(X_values_normalized) 
        factor2 = factor2 - erf(lower_bounds_normalized) 
        factor2 = factor2 - erf(upper_bounds_normalized)
        expectation = factor1 * factor2

        return expectation

    def kappa(self):
        return 1.0

    def iota(self):
        return 1.0
            
    def get_adjusted_gamma(self, gamma):
        return 1.0 if gamma == 'auto' else float(gamma)


class RWPolySign(RKHSWeighting):
    """
    Instantiation of the model using Polynomial kernel and Gaussian distribution, 
    and sign as the base predictor.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    degree : int, default=2
        Degree of the polynomial kernel.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the Model __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input, sigma='auto', degree=2, rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        self.degree = degree
        dist = GaussianDistribution(n_dim=self.n_dim, sigma=sigma, rng=rng)
        kernel = PolynomialKernel(degree=degree)
        base = Sign()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        d = self.degree
        s = self.sigma

        W = self.get_center_params(centers)
        W_norms = self.get_param_norms(centers)
        arr = get_scalar_over_norm(X, W)

        expectation = self.get_empty_expectations(X, centers)
        for j in range(math.floor((d+1)/2) + 1):
            for i in range(j + 1):
                coef = (1 / math.pi) * binom(d, 2*j+1) * binom(2*j+1, 2*i)
                coef *= (2 * s**2)**(j+0.5)
                coef *= sp.special.gamma(j-i+1) * sp.special.gamma(i + 0.5)
                value = np.multiply(arr**(2*j-2*i+1), array_plus_vector(-arr**2, W_norms**2, axis=1)**i)
                expectation = expectation + coef * value

        if len(expectation.flatten()) == 0:
            print(f'W : {W}')
        
        return expectation

    def true_theta(self):
        d = self.degree
        s = self.sigma
        n = self.n_dim
        theta_squared = 0
        for j in range(math.floor((d+1)/2) + 1):
            for i in range(j + 1):
                value = binom(d, 2*j+1) * binom(2*j+1, 2*i)
                value *= (2 * s**2)**(2*j+1) / math.pi**1.5
                value *= sp.special.gamma(j-i+1)**2 * sp.special.gamma(j + 0.5)
                value *= math.exp(sp.special.loggamma(n/2 + j) - sp.special.loggamma((n-1)/2))    
                theta_squared += value
        return math.sqrt(theta_squared)

    def true_kappa(self):
        return self.true_kappa_exact()

    def true_kappa_exact(self):
        d = self.degree
        n = self.n_dim
        sigma = self.sigma
        kappa_squared = 0
        for i in range(self.degree + 1):
            value = binom(d, i)
            value *= (2*sigma**2)**i
            value *= math.exp(sp.special.loggamma(n/2 + i) - sp.special.loggamma(n/2))
            kappa_squared += value
        return math.sqrt(kappa_squared)

    def true_kappa_approx(self):
        """
        Simpler formula, slightly higher value.
        """
        d = self.degree
        n = self.n_dim
        sigma = self.sigma
        kappa_squared = sigma**(2*d) * (n + 2*d + 1/(sigma**2))**d
        return math.sqrt(kappa_squared)

    def true_iota(self):
        return self.kappa()

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)


class RWExpSign(RKHSWeighting):
    """
    Instantiation of the model using Exponential kernel and Gaussian distribution, 
    and sign as the base predictor.

    Parameters
    ----------
    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).

    sigma : {'auto'} or float, default='auto'
        Parameter of the Gaussian distribution.

    gamma : {'auto'} or float, default='auto'
        Parameter of the Gaussian kernel.

    max_theta : float, default=1.5
        Parameter defining the relationship between sigma and gamma, when
        sigma and/or gamma is 'auto'.

    rng : RandomNumberGenerator or int or None, default=None
        The random number generator or randon seed to use.

    use_mc : bool, default=False
        Whether to use Monte Carlo simulations instead of the exact formulas.

    **kwargs
        Additional keyword arguments passed along to the Model __init__.

    Attributes
    ----------
    sigma : float
        The adjusted sigma parameter for the model.

    gamma : float
        The adjusted gamma parameter for the model.

    n_dim : int
        Number of dimensions of the input data.

    dist : Distribution
        Distribution of the parameters. p in the equations.

    kernel : Kernel
        K in the equations

    base : BasePredictor
        phi in the equations
    """
    def __init__(self, input, sigma='auto', gamma='auto', max_theta=1.5, 
                 rng=None, use_mc=False, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        self.max_theta = max_theta
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = GaussianDistribution(n_dim=self.n_dim, sigma=sigma, rng=rng)
        kernel = ExponentialKernel(gamma=gamma)
        base = Sign()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        s = self.sigma
        g = self.gamma

        W = self.get_center_params(centers)
        W_norms = self.get_param_norms(centers)

        cst = s / (g**2 * math.sqrt(8 * math.pi))
        exp_norms = np.exp((cst * W_norms) ** 2)
        arr = get_scalar_over_norm(X, W)

        return array_times_vector(erf(cst * arr), exp_norms, axis=1)

    def true_theta(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 - s**2 / (2*g**2))**(-n/2)

    def true_iota(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 - s**2 / (2*g**2))**(-n/2)

    def true_kappa(self):
        s = self.sigma
        g = self.gamma
        n = self.n_dim
        return (1 - s**2 / g**2)**(-n/4)

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)

    def get_adjusted_gamma(self, gamma):
        if gamma == 'auto':
            return self.get_gamma_from_theta(self.max_theta)
        else:
            return float(gamma)

    def get_gamma_from_theta(self, theta):
        n = self.n_dim
        return self.sigma / math.sqrt(1 - theta**(-4/n))

    def get_gamma_old_formula(self, theta):
        n = self.n_dim
        return math.sqrt(2) * self.sigma / math.sqrt(theta**(-4/n) - 1)