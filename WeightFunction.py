from __future__ import annotations

from copy import deepcopy
from typing import List
import numpy as np
from scipy.linalg import cholesky
import math

from Kernel import *


class Center:
    def __init__(self, param, coef) -> None:
        self.coef = coef
        self.param = param
        try:
            self.param_norm = np.linalg.norm(param)
        except:
            pass


class WeightFunction:
    """
    Class for an element of the RKHS of a given kernel.

    Of the form \sum_{t=1}^T a_t K(w_t, .)

    Parameters
    ----------
    kernel : Kernel,
        Kernel of the RKHS. K in the equations.

    Attributes
    ----------
    kernel : Kernel,
        Kernel of the RKHS. K in the equations.

    list_of_params : List[np.ndarray],
        (w_1, ..., w_T) in the equations.

    coefs : List[float],
        (a_1, ..., a_T) in the equations.

    running_norm : float
        RKHS norm of the weight_function. Might not be up to date.

    norm_is_up_to_date : Bool,
        Whether the running_norm is up to date.
    """
    def __init__(self, kernel: Kernel=GaussianKernel()):
        """
        Initializes the WeightFunction with the specified or default kernel.
        """
        self.kernel = kernel
        self.centers = []
        self.running_norm = 0
        self.norm_is_up_to_date = True
    
    def add_center(self, param, coef) -> WeightFunction:
        """
        Add a center to list_of_params, along with its coefficient.
        """
        if coef != 0:
            if len(self.centers) == 0:
                self.centers.append(Center(param, coef))
                self.running_norm = self.calculate_rkhs_norm()
            else:
                self += WeightFunction(self.kernel).add_center(param, coef)
        return self
    
    def calculate_rkhs_norm(self) -> float:
        """
        Updates and returns the RKHS norm of the weight function.
        """
        self.running_norm = math.sqrt(max(0, self.scalar_product(self)))
        self.norm_is_up_to_date = True
        return self.running_norm

    def scalar_product(self, beta: WeightFunction) -> np.ndarray:
        if self.get_n_centers() == 0 or beta.get_n_centers() == 0:
            return 0.0
        gram = self.kernel.calculate(self.get_center_params(), beta.get_center_params())
        return np.dot(self.get_coefs(), np.dot(gram, beta.get_coefs()))

    def get_n_centers(self) -> int:
        return len(self.centers)
    
    def __add__(self, beta: WeightFunction) -> WeightFunction:
        new = self.copy()
        new.efficient_norm_update(beta)
        new.centers = self.centers + beta.centers
        return new

    def __sub__(self, beta: WeightFunction) -> WeightFunction:
        return self + (-beta)

    def __neg__(self) -> WeightFunction:
        new = self.copy()
        new *= -1
        return new

    def __iadd__(self, beta: WeightFunction) -> WeightFunction:
        self.efficient_norm_update(beta)
        self.centers = self.centers + beta.centers
        return self

    def __rmul__(self, factor) -> WeightFunction:   
        """
        Multiply all coefs by factor, and updates the running norm.
        """
        new = self.copy()
        for center in new.centers:
            center.coef *= factor
        new.running_norm *= factor
        return new

    def __imul__(self, factor) -> WeightFunction:
        """
        Multiply all coefs by factor, and updates the running norm.
        """
        for center in self.centers:
            center.coef *= factor
        self.running_norm *= factor
        return self

    def __truediv__(self, factor) -> WeightFunction:
        """
        Divide all coefs by factor, and updates the running norm.

        Simply returns self if factor is 0.
        """
        if factor != 0:
            for center in self.centers:
                center.coef /= factor
            self.running_norm /= self.running_norm
        return self


    def efficient_norm_update(self, beta: WeightFunction) -> None:
        """
        Updates the running norm given the weight function
        that has newly been added (i.e. the norm of self + beta).
        Much more efficient than self.calculate_rkhs_norm if beta
        has few centers.
        """
        scalar = self.scalar_product(beta)
        norm_squared = self.norm()**2 + 2 * scalar + beta.norm()**2
        try:
            self.running_norm = math.sqrt(norm_squared)
        except ValueError:
            # Due to imprecision, value will sometimes be slightly negative instead of 0.
            self.running_norm = math.sqrt(abs(round(norm_squared, 10)))
        self.norm_is_up_to_date = True
    
    def norm(self) -> float:
        """
        Efficiently returns the RKHS norm of the weight function.
        Will compute only if necessary.
        """
        return self.running_norm if self.norm_is_up_to_date else self.calculate_rkhs_norm()

    def efficient_add(self, beta: WeightFunction, factor=1.0) -> WeightFunction:
        """
        Efficient sum when only the last center is different.
        Yields (self + factor * beta).
        Used for calculating the running average of the model.
        Running RKHS norm is not updated.
        """
        for i, beta_center in enumerate(beta.centers):
            if i < len(self.centers):
                self.centers[i].coef += factor * beta_center.coef
            else:
                self.centers.append(beta_center)
                self.centers[i].coef *= factor 
        self.norm_is_up_to_date = False
        return self

    def update_average(self, beta: WeightFunction, n_iter) -> WeightFunction:
        """
        Assumes self is a running average of n_iter models, and updates
        using the formula :

        self = n_iter / (1 + n_iter) * self + 1 / (1 + n_iter) * beta

        Assumes beta has a single additionnal center.
        """
        self *= n_iter / (1 + n_iter)
        factor = 1 / (1 + n_iter)
        for i, beta_center in enumerate(beta.centers):
            if i < len(self.centers):
                self.centers[i].coef += factor * beta_center.coef
            else:
                self.centers.append(beta_center)
                self.centers[i].coef *= factor 
        self.norm_is_up_to_date = False
        return self

    def copy(self) -> WeightFunction:
        return deepcopy(self)
    
    def set_centers(self, params: List, coefs=None) -> WeightFunction:
        """
        Overwrites the list of centers.
        Used for arbitrarily changing the weight function.
        """
        if coefs is None:
            self.centers = [Center(param, 1) for param in params]
        else:
            self.centers = [Center(param, coef) for param, coef in zip(params, coefs)]
            self.remove_useless_centers()
        self.norm_is_up_to_date = False
        return self

    def set_coefs(self, coefs: List) -> WeightFunction:
        """
        Overwrites the coefficients without changing the centers.
        Used after applying the Lasso.
        """
        for i, center in enumerate(self.centers):
            center.coef = coefs[i]
        self.remove_useless_centers()
        self.norm_is_up_to_date = False
        return self

    def project(self, max_norm) -> float:
        """
        Ensures the RKHS norm is at most max_norm.
        """
        factor = min(1.0, max_norm / self.norm()) if self.norm() else 1.0 
        self *= factor
        return factor
    
    def remove_useless_centers(self) -> WeightFunction:
        """
        Removes centers which have a zero coefficient.
        """
        self.centers = [center for center in self.centers if center.coef != 0]
        return self
    
    def gram(self) -> np.ndarray:
        """
        Returns the Gram matrix of the centers.
        """
        return self.kernel.calculate(self.get_center_params(), self.get_center_params())

    def get_max_center_norm(self) -> float:
        """
        Returns the square root of the maximal value of K(w, w)
        for all w's in the list of centers.
        """
        return math.sqrt(np.max(np.diag(self.gram())))

    def choleski_upper(self) -> np.ndarray:
        n_centers = self.get_n_centers()
        gram = self.gram()
        try:
            choleski_upper = cholesky(gram)
        except:
            choleski_upper = cholesky(gram + 1e-8 * np.identity(n_centers)) 
        return choleski_upper  
    
    def merge_duplicate_centers(self) -> WeightFunction:
        """
        Merges the coefficients for centers that are equal.
        This is a somewhat costly operation.
        Do not use every iteration.
        """
        n = self.get_n_centers()
        if n > 0:
            merged_centers = []
            for center in self.centers:
                if len(merged_centers) == 0:
                    merged_centers.append(center)
                else:
                    for merged_center in merged_centers:
                        if center.param == merged_center.param:
                            merged_center.coef += center.coef
                        else:
                            merged_centers.append(center)
            self.centers = merged_centers
        return self
   
    def eval_weight_func(self, w) -> float:
        beta = WeightFunction(self.kernel).add_center(w, 1)
        return self.scalar_product(beta)

    def eval_weight_func_multiple_centers(self, list_of_w: list) -> np.ndarray:
        return [self.eval_weight_func(w) for w in list_of_w]
    def get_param_norms(self, centers: List[Center] = None):
        centers = self.centers if centers is None else centers
        return np.array([center.param_norm for center in centers])

    def get_center_params(self, centers: List[Center] = None):
        centers = self.centers if centers is None else centers
        W = np.array([center.param for center in centers])
        if W.shape[0] == 0:
            print(centers)
        return np.array([center.param for center in centers])
        
    def get_coefs(self, centers: List[Center] = None) -> np.ndarray:
        centers = self.centers if centers is None else centers
        return np.array([center.coef for center in centers])