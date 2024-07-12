from __future__ import annotations

import numpy as np
import scipy as sp
import seaborn ; seaborn.set()
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from time import time

from Model import RKHSWeighting
from Learners import Learner


class _RKHSWeightingEstimator(BaseEstimator):
    """
    Utility class containing functions shared by RKHSWeightingRegressor and RKHSWeightingClassifier
    """   
    def _set_model(self, new_model: RKHSWeighting):
        """
        Utility function that directly sets self.model to new_model.
        """
        self.model = new_model
    
    def _model_training_loss(self, model: RKHSWeighting) -> np.ndarray:
        check_is_fitted()
        output = model.output(self.data_)
        targets = self.targets_
        return self.learner.loss.calculate(output, targets)
    
    def raw_output(self, X) -> np.ndarray:
        """
        Returns the size m array containing
        the output of the prediction model for all examples in X.

        Equations-wise, this is Lambda alpha(X).
        """
        return self.model.output(X)  


class RKHSWeightingRegressor(_RKHSWeightingEstimator, RegressorMixin):
    def __init__(self, learner: Learner, model: RKHSWeighting) -> None:
        self.learner = learner
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> RKHSWeightingRegressor: 
        X, y = check_X_y(X, y)
        self.model = self.learner.fit_model(X, y, self.model, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model.output(X)
    
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        output = self.raw_output(X)
        return self.learner.loss.calculate(output, y)
    
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted



class RKHSWeightingClassifier(_RKHSWeightingEstimator, ClassifierMixin):
    def __init__(self, learner: Learner, model: RKHSWeighting) -> None:
        self.learner = learner
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> RKHSWeightingClassifier: 
        self._preprocessing(X, y)
        data = self.data_
        targets = self.targets_
        self.model = self.learner.fit_model(data, targets, self.model)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        proba = self.predict_proba(X)
        return self._proba_to_classes(proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba_from_output(self.raw_output(X))  
    
    def _more_tags(self):
        return {'binary_only': True}
    
    def _preprocessing(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        self.data_ = X
        self.classes_, self.targets_ = self._classes_preprocessing(y)

    def _classes_preprocessing(self, y: np.ndarray):
        self.y_ = y
        self.classes_ = np.unique(y)
        self.targets_ = self._classes_to_targets(y)
        return self.classes_, self.targets_

    def _classes_to_targets(self, classes: np.ndarray) -> np.ndarray:
        targets = classes.copy()
        targets[classes == self.classes_[0]] = 1
        targets[classes == self.classes_[1]] = -1
        return targets

    def _targets_to_classes(self, targets: np.ndarray) -> np.ndarray:
        classes = targets.copy()
        classes[targets == 1] = self.classes_[0]
        classes[targets == -1] = self.classes_[1]
        return classes

    def _proba_from_output(self, output: np.ndarray) -> np.ndarray:
        return self._logistic_function(output)

    def _logistic_function(self, array: np.ndarray) -> np.ndarray:
        return sp.special.expit(array) 

    def _proba_to_classes(self, proba: np.ndarray) -> np.ndarray:
        classes = np.zeros(proba.shape, self.classes_.dtype)
        classes[proba > 0.5] = self.classes_[0]
        classes[proba <= 0.5] = self.classes_[1]
        return classes
    
    def _model_training_01_loss(self, model: RKHSWeighting) -> np.ndarray:
        proba = self._proba_from_output(model.output(self.data_))
        pred = self._proba_to_classes(proba)
        return 1 - accuracy_score(self.targets_, pred)
    
    def training_01_loss(self) -> np.ndarray:
        return self._model_training_01_loss(self.model_)
    
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        output = self.raw_output(X)
        targets = self._classes_to_targets(y)
        return self.learner.loss.calculate(output, targets)
    

class RKHSWeightingCV(BaseEstimator):
    def __init__(self, estimator_class, learner_class: Learner, model_class: RKHSWeighting, 
                 learner_param_grid: dict, model_param_grid: dict, folds=5, rng=None, verbose=True) -> None:
        self.estimator_class = estimator_class
        self.learner_class = learner_class
        self.model_class = model_class
        self.learner_param_grid = learner_param_grid
        self.model_param_grid = model_param_grid
        self.folds = folds
        self.rng = rng
        self.verbose = verbose
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> RKHSWeightingCV:
        self._initialize(X, y)
        n_fits = 0
        n_total_fits = len(ParameterGrid(self.learner_param_grid)) * len(ParameterGrid(self.model_param_grid))
        for learner_params in ParameterGrid(self.learner_param_grid):
            for model_params in ParameterGrid(self.model_param_grid):
                start = time()
                self._fit_and_score_one_combination(learner_params, model_params)
                n_fits += 1
                if self.verbose:
                    print(f'Fit {n_fits} of {n_total_fits} done in {time() - start} seconds.')
        self._refit()
        return self
    
    def _initialize(self, X, y):
        self.data_ = X
        self.targets_ = y
        self.best_score_ = 0
        self.best_estimator_ = None
        self.best_learner_params_ = None
        self.best_model_params_ = None

    def _fit_and_score_one_combination(self, learner_params, model_params):
        model = self.model_class(input=self.data_, **model_params, rng=self.rng)
        learner = self.learner_class(**learner_params, rng=self.rng)
        clf = self.estimator_class(learner, model).fit(self.data_, self.targets_)
        score = self.avg_cv_score(clf, self.data_, self.targets_)
        if score > self.best_score_ or self.best_estimator_ is None:
            self.best_score_ = score
            self.best_estimator_ = clf
            self.best_learner_params_ = learner_params
            self.best_model_params_ = model_params

    def _refit(self):
        model = self.model_class(input=self.data_, **self.best_model_params_)
        learner = self.learner_class(model=model, **self.best_learner_params_)
        start_refit = time()
        clf = self.estimator_class(learner, model).fit(self.data_, self.targets_)
        self.refit_time_ = time() - start_refit
        self.best_estimator_ = clf
        
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        return self.best_estimator_.score(X, y)

    def avg_cv_score(self, estimator, X: np.ndarray, y: np.ndarray) -> float:
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0) 
        total_score = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            estimator.fit(X_train, y_train)
            total_score += estimator.score(X_test, y_test)
        return total_score / self.folds