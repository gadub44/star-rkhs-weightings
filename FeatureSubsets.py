from itertools import combinations
import random
from scipy.special import binom as binomial
from typing import List

from Model import *
from Shapley import *
from WeightFunction import Center

import visuals
# visuals.set_visual_settings()

VALID_MODELS = [RWSign, RWRelu, RWPolySign, RWExpSign]
DEFAULT_MODEL = RWSign

VALID_SUBSET_TYPES = ['any', 'seq']
DEFAULT_SUBSET_TYPE = 'seq'

def verify_model_class(model_class):
    if model_class in VALID_MODELS:
        return model_class
    else:
        print(f"Invalid model class. Valid model classes are :{str(VALID_MODELS)}")
        print(f"Using {str(DEFAULT_MODEL)}.")
        return DEFAULT_MODEL
    
def verify_subset_type(subset_type):
    if subset_type in VALID_SUBSET_TYPES:
        return subset_type
    else:
        print(f"Invalid subset type. Valid subset types are :{str(VALID_SUBSET_TYPES)}")
        print(f"Using {str(DEFAULT_SUBSET_TYPE)}.")
        return DEFAULT_SUBSET_TYPE
    
def get_fs_dist(n_dim, fs_length, fs_type, rng):
    if fs_type == 'seq':
        return FSSequenceDistribution(n_dim=n_dim, fs_length=fs_length, rng=rng)
    elif fs_type == 'any':
        return FSArbitraryDistribution(n_dim=n_dim, fs_length=fs_length, rng=rng)
    else:
        raise AttributeError(f"Invalid subset type. Must be in {VALID_SUBSET_TYPES}.")

def adjust_fs_length(fs_length):
    adjusted = [fs_length] if type(fs_length) is int else fs_length
    return adjusted


class FSParam:
    def __init__(self, fs, param):
        self.fs = sorted(list(fs))
        self.param = param

class FSCenter(Center):
    def __init__(self, fs_param, coef):
        self.fs_param = fs_param
        super().__init__(fs_param.param, coef)


class FSArbitraryDistribution(Distribution):
    """
    n_dim : dimensionality of the space
    fs_length : number of features per feature subset. int or list of int

    Generates arbitrary combination of variables
    """
    def __init__(self, n_dim, fs_length, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.fs_length = adjust_fs_length(fs_length)

    def sample(self):
        size = random.choice(self.fs_length)
        return self.rng.choice(self.n_dim, size=size, replace=False)

    def score_samples(self, x):
        """
        x : must be a list of elements
        """
        assert(type(x) == list)
        scores = []
        for fs in x:
            scores.append(self.score_one_sample(fs))
        return scores
    
    def score_one_sample(self, fs):
        size_score = 1 / len(self.fs_length)
        fs_score = 1 / self.n_possible_fs(len(fs))
        return size_score * fs_score
    
    def n_possible_fs(self, length):
        if not hasattr(self, '_n_possible_fs'):
            self._n_possible_fs = {}
            for l in self.fs_length:
                self._n_possible_fs[l] = float(binomial(self.n_dim, l))
        return self._n_possible_fs[length] 
    
    def get_all_valid_fs_generator(self):
        return (combinations(range(self.n_dim), length) for length in self.fs_length)


class FSSequenceDistribution(Distribution):
    """
    n_dim : dimensionality of the space
    fs_length : number of features per feature subset. int or list of int

    Generates sequences of variables.
    """
    def __init__(self, n_dim, fs_length, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.fs_length = adjust_fs_length(fs_length)

    def sample(self):
        size = random.choice(self.fs_length)
        start = self.rng.integers(self.n_possible_fs(size))
        return list(range(start, start + size))

    def score_samples(self, x):
        """
        x : must be a list of elements
        """
        assert(type(x) == list)
        scores = []
        for fs in x:
            scores.append(self.score_one_sample(fs))
        return scores
    
    def score_one_sample(self, fs):
        size_score = 1 / len(self.fs_length)
        fs_score = float(1 / self.n_possible_fs(len(fs)))
        return size_score * fs_score
    
    def n_possible_fs(self, size):
        return 1 + (self.n_dim-size)
    
    def get_all_fs(self):
        for start in range(self.n_dim-self.fs_length):
            yield list(range(start, start + self.fs_length))


class FSParamDistribution(Distribution):
    """
        fs_length : positive integer, or list of positive integers
    """
    def __init__(self, input: np.ndarray, base_model_class=DEFAULT_MODEL, base_model_params={}, 
                 fs_length=1, fs_type='seq', rng=None, **kwargs) -> None:
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.fs_length = adjust_fs_length(fs_length)
        self.fs_type = fs_type
        self.rng = default_rng(rng)
        self.fs_dist = get_fs_dist(self.n_dim, self.fs_length, self.fs_type, self.rng)

    def get_reduced_size_input(self, fs):
        if type(self.input) is int:
            return len(fs)
        elif type(self.input) is np.ndarray:
            return self.input[:, fs]

    def sample(self):
        # Sample the FS
        fs = self.fs_dist.sample()

        # Make the base model
        input = self.get_reduced_size_input(fs)
        base_model = self._get_base_model_from_input(input)

        # Sample the parameter
        param = base_model.sample_center()

        return FSParam(fs, param)

    def score_samples(self, x: List[FSParam]):
        """
        x : must be a list of Center objects
        """
        scores = []
        for center in x:
            scores.append(self.score_one_sample(center))
        return np.array(scores)
    
    def score_one_sample(self, fs_param: FSParam):
        fs = fs_param.fs
        param = fs_param.param

        # Make the base model
        input = self.get_reduced_size_input(fs)
        base_model = self._get_base_model_from_input(input)

        # Compute scores
        size_score = 1 / len(self.fs_length)
        fs_score = self.fs_dist.score_one_sample(fs)
        param_score = float(base_model.dist.score_samples([param]))

        return float(size_score * fs_score * param_score)
    
    def _get_base_model_from_input(self, input) -> RKHSWeighting:
        return self.base_model_class(input=input, rng=self.rng, **self.base_model_params)


class FeatureIndicatorKernel(Kernel):
    def __init__(self) -> None:
        pass
    
    def calculate(self, W: List[List[int]], U: List[List[int]]) -> np.ndarray:
        W_sorted = [sorted(w) for w in W]
        U_sorted = [sorted(u) for u in U]
        G = np.zeros(shape=(len(W_sorted), len(U_sorted)), dtype=int)
        for i, w in enumerate(W_sorted):
            for j, u in enumerate(U_sorted):
                if w == u:
                    G[i, j] = 1
        return G
    
    
class FSParamKernel(Kernel):
    def __init__(self, fs_kernel: Kernel, param_kernel: Kernel) -> None:
        self.fs_kernel = fs_kernel
        self.param_kernel = param_kernel
    
    def calculate(self, W: List[FSParam], U: List[FSParam]) -> np.ndarray:
        W_fs = [center.fs for center in W]
        W_params = [center.param for center in W]
        U_fs = [center.fs for center in U]
        U_params = [center.param for center in U]
        G_FS = self.fs_kernel.calculate(W_fs, U_fs)
        n_w = len(W)
        n_u = len(U)
        G = np.zeros((n_w, n_u))
        for i in range(n_w):
            for j in range(n_u):
                fs_kernel_value = G_FS[i, j]
                if fs_kernel_value:
                    param_kernel_value = self.param_kernel.calculate([W_params[i]], [U_params[j]])
                    G[i,j] = fs_kernel_value * param_kernel_value
        return G


class FSPredictor(BasePredictor):
    def __init__(self, base_pred: BasePredictor) -> None:
        self.base_pred = base_pred
    
    def eval(self, W: List[FSParam], X: np.ndarray) -> np.ndarray:
        """
        Return the n_examples x n_parameters array of the
        base predictor on every pair (example, parameter).
        """
        m = X.shape[0]
        T = len(W)
        preds = np.zeros(shape=(m, T))
        for j, center in enumerate(W):
            fs = center.fs
            w = center.param
            w_slice = w if len(w) == len(fs) else w[fs]
            x_slice = X[:, fs]
            value = self.base_pred.eval(w_slice, x_slice)
            preds[:, j] = value
        return preds


class FSModel(RKHSWeighting):
    def __init__(self, input: np.ndarray, base_model_class=DEFAULT_MODEL, base_model_params={}, 
                 fs_length=1, fs_type='seq', rng=None, use_mc=False, **kwargs) -> None:
        params = locals()
        params.pop('self')
        self.input = input
        self.n_dim = get_n_dim_from_input(input)
        self.fs_length = adjust_fs_length(fs_length)
        self.fs_type = fs_type
        self.data = input
        self.rng = default_rng(rng)
        self.base_model_class = verify_model_class(base_model_class)
        self.base_model_params = base_model_params
        self.full_dim_base_model = base_model_class(input=input, rng=self.rng, **base_model_params)
        base_pred = FSPredictor(self.full_dim_base_model.base_pred)
        kernel = FSParamKernel(fs_kernel=FeatureIndicatorKernel(), param_kernel=self.full_dim_base_model.kernel)
        dist = FSParamDistribution(**params)
        super().__init__(dist, kernel, base_pred, self.rng, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X: np.ndarray, centers: List[Center] = None) -> np.ndarray:
        """
        Returns the (n_examples, n_centers) array containing
        the value of the expectation for each (center, x) pair.
        """
        centers = self.centers if centers is None else centers
        m = X.shape[0]
        T = len(centers)
        expects = np.zeros(shape=(m, T))
        for j, center in enumerate(centers):
            if len(self._expectations_for_one_center(X, center).flatten()) == 0:
                print(center.param.fs)
                print(self._expectations_for_one_center(X, center))
                print(self._expectations_for_one_center(X, center).flatten())
            expects[:, j] = self._expectations_for_one_center(X, center).flatten()
        return expects
    
    def _expectations_for_one_center(self, X: np.ndarray, center: Center) -> np.ndarray:
        """
        Calculates the expectation for one center.
        """
        fs = center.param.fs
        w = center.param.param
        X_eff = X[:, fs]
        w_eff = w if len(w) == len(fs) else w[fs]
        prob = self.dist.fs_dist.score_one_sample(fs)
        model = self.get_base_model(X_eff)
        center = Center(w_eff, 1)
        expects = model.expectations(X_eff, [center])
        if len(prob * expects.flatten()) == 0:
            print(prob, expects, prob * expects, f'center : {center.param}, {center.coef}')
        return prob * expects
    
    def get_base_model(self, X_eff: np.ndarray):
        """
            Returns (makes if necessary) a base model of the adequate dimensionality.
        """
        n_dim = X_eff.shape[1]
        if not hasattr(self, 'base_model_dict'):
            self.base_model_dict = {}
        if n_dim not in self.base_model_dict.keys():
            self.base_model_dict[n_dim] = self.base_model_class(input=X_eff, rng=self.rng, **self.base_model_params)
        return self.base_model_dict[n_dim]
    
    def theta(self):
        """
            theta^2 <= E_{fs} p(fs)theta_{fs}^2
        """
        if len(self.centers) == 0:
            return 0.0
        else:
            thetas_squared = []
            p_fs = []
            for center in self.centers:
                thetas_squared.append(self._theta_for_one_center(center)**2)
                p_fs = self.dist.fs_dist.score_one_sample(center.param.fs)
            return float(np.sqrt(np.mean(np.multiply(p_fs, thetas_squared))))
    
    def _theta_for_one_center(self, center: Center):
        input = self.dist.get_reduced_size_input(center.param.fs)
        base_model = self.dist._get_base_model_from_input(input)
        return base_model.theta()
    
    def iota(self):
        if len(self.centers) == 0:
            return 0.0
        else:
            iotas = []
            for center in self.centers:
                iotas.append(self._iota_for_one_center(center))
            return float(np.mean(iotas))
    
    def _iota_for_one_center(self, center: Center):
        input = self.dist.get_reduced_size_input(center.param.fs)
        base_model = self.dist._get_base_model_from_input(input)
        return base_model.iota()
    
    def kappa(self):
        if len(self.centers) == 0:
            return 0.0
        else:
            kappas_squared = []
            for center in self.centers:
                kappas_squared.append(self._kappa_for_one_center(center)**2)
            return float(np.sqrt(np.mean(kappas_squared)))
    
    def _kappa_for_one_center(self, center: Center):
        input = self.dist.get_reduced_size_input(center.param.fs)
        base_model = self.dist._get_base_model_from_input(input)
        return base_model.kappa()
    
    def partial(self, fs):
        """
        Returns the partial model consisting of only the terms
        with the given feature subset.
        """
        sorted_fs = sorted(fs)
        centers = [center for center in self.centers if center.param.fs == sorted_fs]
        def partial(X):
            return self.output(X, centers)
        return partial
    
    def get_all_unique_fs(self):
        unique_fs = []
        for fs in [sorted(list(center.param.fs)) for center in self.centers]:
            if fs not in unique_fs:
                unique_fs.append(fs)
        return unique_fs