import numpy as np
import os

def normalize_data(X: np.ndarray):
    """
    Assumes X is a 2d array.

    Return the array where all lines now have euclidean norm 1.

    Null lines are unaffected.
    """
    X_norms = np.linalg.norm(X, axis=1)
    return array_times_vector(X[X_norms != 0, :], 1 / X_norms[X_norms != 0], axis=0)

def get_scalar_over_norm(X, W):
    """
        Returns <w, x> / ||x||.
    """
    X_norms = np.linalg.norm(X, axis=1)
    X_norms[X_norms == 0] = 1
    return array_times_vector(np.dot(W, X.T), 1.0 / X_norms, axis=1).T

def array_plus_vector(arr: np.ndarray, vec: np.ndarray, axis) -> np.ndarray:
    m, n = arr.shape
    vec = vec.flatten()
    if axis == 1:
        assert(len(vec) == n)
        return arr + vec
    elif axis == 0:
        assert(len(vec) == m)
        return (arr.T + vec).T

def array_times_vector(arr: np.ndarray, vec: np.ndarray, axis) -> np.ndarray:
    m, n = arr.shape
    vec = vec.flatten()
    if axis == 1:
        assert(len(vec) == n)
        return np.multiply(arr, vec)
    elif axis == 0:
        assert(len(vec) == m)
        return (arr.T * vec).T

def get_n_dim_from_input(input) -> int:
    if type(input) is int:
        n_dim = input
    elif type(input) is np.ndarray:
        n_dim = input.shape[1]
    else:
        print("Invalid 'input' : {}".format(input))
    return n_dim

def ensure_X_2d(X: np.ndarray):
    """
    Ensures that X is a two dimensional array
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    return X

def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)