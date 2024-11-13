from itertools import combinations, permutations
from math import factorial
import numpy as np
from scipy.special import comb
from typing import List
from tqdm import tqdm

from utils import ensure_X_2d

def replace_over_FS(data: np.ndarray, x: np.ndarray, FS: List[int]):
    if len(FS) == 0:
        return data
    else:
        data_copy = data.copy()
        data_copy[:, FS] = x[FS]
        return data_copy

def generate_all_As_positive(idx: int, FS: List[int]):
    assert(idx in FS)
    # idx is always in A
    A = [idx]
    FS_minus_idx = list(set(FS).difference([idx]))

    # A can have size 1 to |FS|-1 (there must be at least one element not in A)
    return (
        A + list(sublist)
        for length in range(0, len(FS_minus_idx))
        for sublist in combinations(FS_minus_idx, length)
    )

def generate_all_As_negative(idx: int, FS: List[int]):
    assert(idx in FS)
    # idx is always in A
    A = [idx]
    FS_minus_idx = list(set(FS).difference([idx]))

    # A can have size 2 to |FS|
    return (
        A + list(sublist)
        for length in range(1, len(FS_minus_idx)+1)
        for sublist in combinations(FS_minus_idx, length)
    )

def all_nonempty_subsets(fs):
    for size in range(1, len(fs)+1):
        for subset in combinations(fs, size):
            yield list(subset)

def is_subset(small, big):
    return set(small).issubset(set(big))

def in_Ap_idx_fs(idx, fs, A):
    return is_subset(A, fs) and idx in A and len(A) >= 1 and len(A) < len(fs)

def in_An_idx_fs(idx, fs, A):
    return is_subset(A, fs) and idx in A and len(A) > 1 and len(A) <= len(fs)
    

class Explainer:
    def __init__(self, model, data: np.ndarray) -> None:
        self.model = model
        self.data = data

    def single_shap_value(self, idx: int, X: np.ndarray):
        """
            Calculates the Shapley value of variable idx for example x.
        """
        raise NotImplementedError
    
    def shap_values(self, X: np.ndarray):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        values = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                values[i, j] = self.single_shap_value(j, X[i, :])
        return values.squeeze().tolist()
    
    def feature_importance(self):
        values = self.shap_values(self.data)
        return np.mean(np.abs(values), axis=0)


class GAM_Explainer(Explainer):
    """
        Explainer for Generalized Additive Models.

        Inefficient. Use shap.AdditiveExplainer instead.    
    """
    def single_shap_value(self, idx: int, x: np.ndarray):
        all_but_idx = list(set(list(range(len(x)))).difference([idx]))
        example_contribution = self.model(x.reshape(1, -1))
        background_contribution = np.mean(self.model(replace_over_FS(self.data, x, all_but_idx)))
        return float(example_contribution - background_contribution)


class STAR_Explainer(Explainer):
    """
        Class for calculating the Shapley values of a
        Structured Additive Regression Model (STAR).

        The given model must implement:
          model.partial(FS)         : returns the partial model defined on the feature subset FS
          model.get_all_unique_fs() : returns all feature subsets used to define the model
    """
    
    def shap_values(self, X: np.ndarray, verbose=False):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        values = np.zeros(X.shape)
        for m in tqdm(range(X.shape[0]), delay=2):
            if verbose:
                print(f'Calculating for example {m} of {X.shape[0]}.')
            x = X[m, :]
            for fs in self.model.get_all_unique_fs():
                card_fs = len(fs)
                partial_model = self.model.partial(fs)
                baseline = (partial_model(x) - np.mean(partial_model(self.data))) / card_fs
                for idx in fs:
                    values[m, idx] += baseline
                for A in all_nonempty_subsets(fs):
                    card_A = len(A)
                    coef = card_A * comb(card_fs, card_A)
                    replaced = replace_over_FS(self.data, x, A)
                    term = np.mean(partial_model(replaced))
                    for idx in A:
                        if in_Ap_idx_fs(idx, fs, A):
                            values[m, idx] += term / coef
                    FS_minus_A = list(set(fs).difference(set(A)))
                    for idx in FS_minus_A:
                        A_with_idx = A + [idx]
                        card_A = len(A_with_idx)
                        coef = card_A * comb(card_fs, card_A)
                        if in_An_idx_fs(idx, fs, A_with_idx):
                            values[m, idx] -= term / coef
        if X.shape[0] == 1:
            return values.squeeze().tolist()
        else:
            return values.tolist()
            
    
    def single_shap_value(self, idx: int, x: np.ndarray):
        return self.shap_values(x)[idx]
    

class STAR_ExplainerUnoptimized(Explainer):
    """
        Deprecated, but useful for testing.
        Implements a less optimized version of the formula.

        Class for calculating the Shapley values of a
        Structured Additive Regression Model (STAR).

        The given model must implement:
          model.partial(FS)         : returns the partial model defined on the feature subset FS
          model.get_all_unique_fs() : returns all feature subsets used to define the model
    """
    def positive_expectation(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        first = self.positive_expectation_first_term(FS, x)
        second = self.positive_expectation_second_term(idx, FS, x)
        return first + second
    
    def positive_expectation_first_term(self, FS: List[int], x: np.ndarray):
        return float(self.model.partial(FS)(x) / len(FS))
    
    def positive_expectation_second_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        partial_model = self.model.partial(FS)
        card_fs = len(FS)
        value = 0
        for A in generate_all_As_positive(idx, FS):
            card_A = len(A)
            coef = 1 / (card_A * comb(card_fs, card_A))
            value += coef * np.mean(partial_model(replace_over_FS(self.data, x, A)))
        return float(value)
    
    def brute_force_positive_expectation_first_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        value = 0
        partial_model = self.model.partial(FS)
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx_with_idx = p[:p.index(idx)+1]
            if set(FS).issubset(set(up_to_idx_with_idx)):
                value += np.mean(partial_model(replace_over_FS(self.data, x, up_to_idx_with_idx)))
        return value / factorial(len(x))
    
    def brute_force_positive_expectation_second_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        value = 0
        partial_model = self.model.partial(FS)
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx_with_idx = p[:p.index(idx)+1]
            if not set(FS).issubset(set(up_to_idx_with_idx)):
                value += np.mean(partial_model(replace_over_FS(self.data, x, up_to_idx_with_idx)))
        return value / factorial(len(x))

    def negative_expectation(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        first = self.negative_expectation_first_term(FS)
        second = self.negative_expectation_second_term(idx, FS, x)
        return first + second
    
    def negative_expectation_first_term(self, FS: List[int]):
        return float(np.mean(self.model.partial(FS)(self.data)) / len(FS))

    def negative_expectation_second_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        partial_model = self.model.partial(FS)
        card_fs = len(FS)
        value = 0
        for A in generate_all_As_negative(idx, FS):
            card_A = len(A)
            coef = 1 / (card_A * comb(card_fs, card_A))
            A_minus_idx = list(set(A).difference([idx]))
            value += coef * np.mean(partial_model(replace_over_FS(self.data, x, A_minus_idx)))
        return float(value)
    
    def brute_force_negative_expectation_first_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        value = 0
        partial_model = self.model.partial(FS)
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx = p[:p.index(idx)]
            if set(FS).isdisjoint(set(up_to_idx)):
                value += np.mean(partial_model(replace_over_FS(self.data, x, up_to_idx)))
        return value / factorial(len(x))
    
    def brute_force_negative_expectation_second_term(self, idx: int, FS: List[int], x: np.ndarray):
        assert(idx in FS)
        value = 0
        partial_model = self.model.partial(FS)
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx = p[:p.index(idx)]
            if not set(FS).isdisjoint(set(up_to_idx)):
                value += np.mean(partial_model(replace_over_FS(self.data, x, up_to_idx)))
        return value / factorial(len(x))
    
    def single_shap_value(self, idx, x: np.ndarray):
        total_positive_expectation = 0
        total_negative_expectation = 0
        for fs in self.model.get_all_unique_fs():
            if idx in fs:
                total_positive_expectation += self.positive_expectation(idx, fs, x)
                total_negative_expectation += self.negative_expectation(idx, fs, x)
        shapley_value = total_positive_expectation - total_negative_expectation
        return shapley_value


class BruteForceExplainer(Explainer):    
    def single_shap_value(self, idx: int, x: np.ndarray): 
        value = 0
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx = p[:p.index(idx)]
            up_to_idx_with_idx = p[:p.index(idx)+1]
            value += np.mean(self.model(replace_over_FS(self.data, x, up_to_idx_with_idx)))
            value -= np.mean(self.model(replace_over_FS(self.data, x, up_to_idx)))
        return value / factorial(len(x))
    
    def positive_expectation(self, idx: int, x: np.ndarray):
        value = 0
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx_with_idx = p[:p.index(idx)+1]
            value += np.mean(self.model(replace_over_FS(self.data, x, up_to_idx_with_idx)))
        return value / factorial(len(x))
    
    def negative_expectation(self, idx: int, x: np.ndarray):
        value = 0
        for p in permutations(range(len(x))):
            p = list(p)
            up_to_idx = p[:p.index(idx)]
            value += np.mean(self.model(replace_over_FS(self.data, x, up_to_idx)))
        return value / factorial(len(x))
    
class SOUM_term:
    def __init__(self, fs, coef) -> None:
        self.fs = sorted(fs)
        self.coef = coef

class SOUM:
    """
        Sum of unanimity model.

        X must be a vector of 0's and 1's.
    """
    def __init__(self) -> None:
        self.terms = []

    def add_term(self, term: SOUM_term):
        self.terms.append(term)

    def partial(self, fs):
        relevant_terms = [term for term in self.terms if term.fs == fs]
        def partial_model(X: np.ndarray):
            X_2d = ensure_X_2d(X)
            output = np.zeros(X_2d.shape[0])
            for term in relevant_terms:
                output += term.coef * (X_2d[:, fs].reshape(X_2d.shape[0], len(fs)) == 1).all(axis=1).astype(int)
            return output
        return partial_model
    
    def get_all_unique_fs(self):
        unique_fs = []
        for fs in [sorted(list(term.fs)) for term in self.terms]:
            if fs not in unique_fs:
                unique_fs.append(fs)
        return unique_fs

    def __call__(self, X: np.ndarray) -> float:
        output = np.zeros(X.shape[0])
        for fs in self.get_all_unique_fs():
            output += self.partial(fs)(X)
        return output
    

class SOUM_Generator:
    def __init__(self, n_dim, n_terms, min_fs_length, max_fs_length, rng=0) -> None:
        self.n_dim = n_dim
        self.n_terms = n_terms
        self.min_fs_length = min_fs_length
        self.max_fs_length = max_fs_length
        self.rng = np.random.default_rng(rng)

    def sample(self) -> SOUM:
        soum = SOUM()
        for _ in range(self.n_terms):
            fs_length = self.rng.integers(low=self.min_fs_length, high=self.max_fs_length+1)
            fs = self.rng.choice(self.n_dim, size=fs_length, replace=False)
            coef = self.rng.uniform()
            soum.add_term(SOUM_term(fs, coef))
        return soum
    
def generate_boolean_dataset(n_examples, n_dim, rng=0):
    rng = np.random.default_rng(rng)
    return rng.choice([0,1], size=(n_examples, n_dim))