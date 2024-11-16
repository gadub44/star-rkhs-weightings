import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shap
import shapiq

from expe_utils import RESULTS_FOLDER, FIGURES_FOLDER
from Shapley import STAR_Explainer, SOUM_Generator, generate_boolean_dataset

import visuals
visuals.set_visual_settings()

parser = argparse.ArgumentParser(description='Computation time comparison of STAR_Explainer.')
parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the figure.')
parser.add_argument('--test', action='store_true', help='Quicker experiment for testing purposes.')
args = parser.parse_args()

# TODO Make the figures an average

RNG = np.random.default_rng(0)
VERBOSE = True
RUN = not args.norun
if args.test:
    DIMS = list(range(2, 20, 1)) + list(range(20, 201, 10))
    MAX_FS_LENGTH = 4
    MIN_FS_LENGTH = MAX_FS_LENGTH
    N_SAMPLES = 10 # Increase this to smooth out the figure
    MAX_TIME = 3
    RESULTS_PATH = RESULTS_FOLDER + 'shapley_time-test.csv'
    FIG_PATH_LEFT = FIGURES_FOLDER + 'shapley_time_left-test.pdf'
    FIG_PATH_RIGHT = FIGURES_FOLDER + 'shapley_time_right-test.pdf'
else:
    DIMS = list(range(2, 20, 1)) + list(range(20, 201, 10))
    MAX_FS_LENGTH = 5
    MIN_FS_LENGTH = MAX_FS_LENGTH
    N_SAMPLES = 100 # Increase this to smooth out the figure
    MAX_TIME = 30 # 5 minutes
    RESULTS_PATH = RESULTS_FOLDER + 'shapley_time.csv'
    FIG_PATH_LEFT = FIGURES_FOLDER + 'shapley_time_left.pdf'
    FIG_PATH_RIGHT = FIGURES_FOLDER + 'shapley_time_right.pdf'

def STAR_time(model, X):
    explainer = STAR_Explainer(model, X)
    start = time()
    star_values = explainer.shap_values(X)
    return time()-start, star_values

def SHAP_time(model, X):
    explainer = shap.explainers.Exact(model, X)
    start = time()
    shap_values = explainer(X).values
    return time()-start, shap_values

def shapiq_values_for_approximator(model, X: np.ndarray, approximator):
    start = time()
    explainer = shapiq.TabularExplainer(
        model=model,
        data=X,
        approximator=approximator,
        imputer="marginal",
        index="SV",
        max_order=1,
        sample_size=X.shape[0],
        random_state=42,
    )
    shap_values = explainer.explain_X(X, budget=256)
    return time()-start, shap_values 

def KSHAP_time(model, X: np.ndarray):
    approximator = shapiq.approximator.KernelSHAP(n=X.shape[1])
    return shapiq_values_for_approximator(model, X, approximator)

def UKSHAP_time(model, X: np.ndarray):
    approximator = shapiq.approximator.UnbiasedKernelSHAP(n=X.shape[1])
    return shapiq_values_for_approximator(model, X, approximator)

def SHAPIQ_time(model, X: np.ndarray):
    approximator = shapiq.approximator.SHAPIQ(n=X.shape[1], index="SV", max_order=1)
    return shapiq_values_for_approximator(model, X, approximator)

def run_expe():
    df = pd.DataFrame()
    run_nstar = True
    run_shap = True
    for n in DIMS:
        X = generate_boolean_dataset(N_SAMPLES, n)
        # STAR-SHAP with fs_length=n
        if run_nstar:
            soum_generator = SOUM_Generator(n_dim=n, n_terms=50, min_fs_length=n, max_fs_length=n)
            model = soum_generator.sample()
            star_time, star_values = STAR_time(model, X)
            star_results = pd.DataFrame({'Algorithm' : ['STAR-SHAP (n-STAR)'], 'n' : [n], 'time' : [star_time]})
            if VERBOSE:
                print(f'STAR time for FS size {n} and n={n}  : {round(star_time, 1)} seconds.')
            df = pd.concat([df, star_results], ignore_index=True)
            if star_time > MAX_TIME:
                run_nstar = False
        # STAR-SHAP with fs_length=k
        for k in range(MIN_FS_LENGTH, MAX_FS_LENGTH+1):
            if k <= n:
                soum_generator = SOUM_Generator(n_dim=n, n_terms=50, min_fs_length=k, max_fs_length=k)
                model = soum_generator.sample()
                star_time, star_values = STAR_time(model, X)
                star_results = pd.DataFrame({'Algorithm' : [f'STAR-SHAP ({k}-STAR)'], 'n' : [n], 'time' : [star_time]})
                if VERBOSE:
                    print(f'STAR time for FS size {k} and n={n}  : {round(star_time, 1)} seconds.')
                df = pd.concat([df, star_results], ignore_index=True)
        # SHAP
        if run_shap:
            shap_time, shap_values = SHAP_time(model, X)  
            shap_results = pd.DataFrame({'Algorithm' : ['SHAP'], 'n' : [n], 'time' : [shap_time]})
            df = pd.concat([df, shap_results], ignore_index=True)
            if VERBOSE:
                print(f'SHAP time n={n} : {round(shap_time, 1)} seconds.')
            if not np.allclose(star_values, shap_values):
                print(f'STAR and SHAP disagree : \n{star_values} \nversus \n{shap_values}')
                raise AssertionError
            if shap_time > MAX_TIME:
                run_shap = False
        # Kernel SHAP
        shap_time, shap_values = KSHAP_time(model, X)  
        shap_results = pd.DataFrame({'Algorithm' : ['Kernel SHAP'], 'n' : [n], 'time' : [shap_time]})
        df = pd.concat([df, shap_results], ignore_index=True)
        if VERBOSE:
            print(f'Kernel SHAP time n={n} : {round(shap_time, 1)} seconds.')
        # Unbiased Kernel SHAP
        shap_time, shap_values = UKSHAP_time(model, X)  
        shap_results = pd.DataFrame({'Algorithm' : ['Unbiased Kernel SHAP'], 'n' : [n], 'time' : [shap_time]})
        df = pd.concat([df, shap_results], ignore_index=True)
        if VERBOSE:
            print(f'Unbiased Kernel SHAP time n={n} : {round(shap_time, 1)} seconds.')
        # SHAP-IQ
        shap_time, shap_values = SHAPIQ_time(model, X)  
        shap_results = pd.DataFrame({'Algorithm' : ['SHAP-IQ'], 'n' : [n], 'time' : [shap_time]})
        df = pd.concat([df, shap_results], ignore_index=True)
        if VERBOSE:
            print(f'SHAP-IQ time n={n} : {round(shap_time, 1)} seconds.')
            
    df.to_csv(RESULTS_PATH)

def make_fig(which='left'):
    df = pd.read_csv(RESULTS_PATH)
    grouped = df.groupby('Algorithm')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray', zorder=1)
    MARKERS = ['+', 'x', '*', 'v', '1', 'D']
    COLORS = plt.cm.tab10.colors  # Choose a colormap or fixed colors
    algo_styles = {algo: (marker, color) for algo, marker, color in zip(grouped.groups.keys(), MARKERS, COLORS)}

    i=0
    np_dims = np.array(DIMS)
    idx_20 = np.argmax(np_dims == 20)
    for algo, group in grouped:
        max_dim = max(group['n'])
        marker, color = algo_styles[algo]
        # Divide the two figures at n_dims = 20
        if max_dim >= 20:
            idx_20 = np.argmax(group['n'] == 20)
        else:
            idx_20 = len(group['n'])
        if which == 'left':
            plt.plot(group['n'][1:idx_20], group['time'][1:idx_20], 
                    label=algo, color=color, linewidth=0.8, antialiased=True, zorder=3,
                    marker=marker, markersize=8)
        elif which == 'right':
            plt.plot(group['n'][idx_20:], group['time'][idx_20:], 
                    label=algo if idx_20 < len(group['n']) else '_nolegend_',
                    color=color, linewidth=0.8, antialiased=True, zorder=3,
                    marker=marker, markersize=8)
        i += 1
    ax.set_axisbelow(False)
    ax.set_xlabel('Dimensionality (n)', fontsize=14)
    plt.ylabel('Time (s)', fontsize=14, labelpad=10)
    plt.legend(frameon=True, fontsize=12, facecolor='white', framealpha=1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    if which == 'left':
        plt.savefig(FIG_PATH_LEFT, bbox_inches='tight')
    if which == 'right':
        plt.savefig(FIG_PATH_RIGHT, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    if RUN:
        run_expe()
    make_fig('left')
    make_fig('right')