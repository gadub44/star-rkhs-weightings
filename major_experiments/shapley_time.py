import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shap

from FeatureSubsets import FSModel
from Learners import LeastSquaresLearner
from RKHSWeightings import RKHSWeightingRegressor
from Shapley import STAR_Explainer
from sklearn.datasets import make_regression

import visuals
visuals.set_visual_settings()

parser = argparse.ArgumentParser(description='Computation time comparison of SAM_Explainer.')
parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the figure.')
args = parser.parse_args()

RNG = np.random.default_rng(0)
MAX_DIM = 20
MAX_FS_LENGTH = 5
MIN_FS_LENGTH = MAX_FS_LENGTH
MAX_SHAP_DIM = 13
MAX_N_SHAP_DIM = 8
N_SAMPLES = 100 # Increase this to smooth out the figure
RESULTS_PATH = './results/shapley_time.csv'
FIG_PATH = './figures/shapley_time.pdf'
VERBOSE = True
RUN = not args.norun

def STAR_time(model, X):
    sam_explainer = STAR_Explainer(model, X)
    start = time()
    star_values = sam_explainer.shap_values(X)
    return time()-start, star_values

def SHAP_time(model, X):
    shap_explainer = shap.explainers.Exact(model, X)
    start = time()
    shap_values = shap_explainer(X).values
    return time()-start, shap_values

def STAR_results(X, y, k, algo_name):
    model = FSModel(input=X, fs_length=k, fs_type='any', rng=RNG)
    learner = LeastSquaresLearner(n_iter=100, rng=RNG)
    clf = RKHSWeightingRegressor(learner, model).fit(X, y)
    star_time, star_values = STAR_time(clf.model, X)
    star_results = pd.DataFrame({'Algorithm' : [algo_name], 'n' : [n], 'time' : [star_time]})
    if VERBOSE:
        print(f'STAR time for FS size {k} and n={n}  : {round(star_time, 1)} seconds.')
    return star_results, clf.model

if __name__ == '__main__':
    if RUN:
        df = pd.DataFrame()
        for n in range(1, MAX_DIM+1):
            X, y = make_regression(n_samples=N_SAMPLES, n_features=n)
            # n-SHAP
            if n <= MAX_N_SHAP_DIM:
                model = FSModel(input=X, fs_length=n, fs_type='any', rng=RNG)
                learner = LeastSquaresLearner(n_iter=100, rng=RNG)
                clf = RKHSWeightingRegressor(learner, model).fit(X, y)
                star_time, star_values = STAR_time(clf.model, X)
                star_results = pd.DataFrame({'Algorithm' : ['n-STAR'], 'n' : [n], 'time' : [star_time]})
                if VERBOSE:
                    print(f'STAR time for FS size {n} and n={n}  : {round(star_time, 1)} seconds.')
                df = pd.concat([df, star_results], ignore_index=True)
            # k-SHAP
            for k in range(MIN_FS_LENGTH, MAX_FS_LENGTH+1):
                if k <= n:
                    model = FSModel(input=X, fs_length=k, fs_type='any', rng=RNG)
                    learner = LeastSquaresLearner(n_iter=100, rng=RNG)
                    clf = RKHSWeightingRegressor(learner, model).fit(X, y)
                    star_time, star_values = STAR_time(clf.model, X)
                    star_results = pd.DataFrame({'Algorithm' : [f'{k}-STAR'], 'n' : [n], 'time' : [star_time]})
                    if VERBOSE:
                        print(f'STAR time for FS size {k} and n={n}  : {round(star_time, 1)} seconds.')
                    df = pd.concat([df, star_results], ignore_index=True)
            # SHAP
            if n <= MAX_SHAP_DIM:
                shap_time, shap_values = SHAP_time(clf.model, X)  
                shap_results = pd.DataFrame({'Algorithm' : ['SHAP'], 'n' : [n], 'time' : [shap_time]})
                df = pd.concat([df, shap_results], ignore_index=True)
                if VERBOSE:
                        print(f'SHAP time n={n} : {round(shap_time, 1)} seconds.')
                if not np.allclose(star_values, shap_values):
                    print(f'STAR and SHAP disagree : \n{star_values} \nversus \n{shap_values}')
                    raise AssertionError
                
        df.to_csv(RESULTS_PATH)

    df = pd.read_csv(RESULTS_PATH)
    grouped = df.groupby('Algorithm')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray', zorder=1)
    for algo, group in grouped:
        if algo == 'SHAP':
            plt.plot(group['n'][1:12], group['time'][1:12], label=algo, linewidth=2, antialiased=True, zorder=3)
        else:
            plt.plot(group['n'], group['time'], label=algo, linewidth=2, antialiased=True, zorder=3)
    ax.set_axisbelow(False)
    ax.set_xticks(range(0, MAX_DIM+1, 2))
    ax.set_xlabel('n', fontsize=14)
    plt.ylabel('time (s)', fontsize=14, labelpad=10)
    plt.legend(frameon=True, fontsize=12, facecolor='white', framealpha=1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(FIG_PATH, bbox_inches='tight')
    plt.show()
