import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shap
import shapiq
from sklearn.metrics import r2_score

from expe_utils import RESULTS_FOLDER, FIGURES_FOLDER
from Shapley import STAR_Explainer
from Shapley import SOUM_Generator, generate_boolean_dataset

import visuals
visuals.set_visual_settings()

parser = argparse.ArgumentParser(description='Variance of the SHAP-IQ approximation of the Shapley Values.')
parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the figure.')
parser.add_argument('--test', action='store_true', help='Quicker experiment for testing purposes.')
args = parser.parse_args()

RNG = np.random.default_rng(0)
VERBOSE = True
RUN = not args.norun
if args.test:
    DIMS = [500]
    N_TERMS = [500]
    MAX_FS_LENGTHS = [8]
    BUDGETS = [np.linspace(10, 5000, num=5, dtype=int)]
    MIN_FS_LENGTH = 8
    N_SAMPLES = 2
    N_SHAPIQ_RUNS = 2
else:
    # DIMS = [10, 100, 500]
    DIMS = [10, 500]
    # N_TERMS = [100, 100, 100]
    N_TERMS = [100, 100]
    # MAX_FS_LENGTHS = [4, 7, 10]
    MAX_FS_LENGTHS = [4, 10] # 7
    BUDGETS = [np.linspace(10, 500, num=20, dtype=int),
            #    np.linspace(10, 2000, num=20, dtype=int),
               np.linspace(10, 10000, num=20, dtype=int),]   
    MIN_FS_LENGTH = 2
    N_SAMPLES = 10
    N_SHAPIQ_RUNS = 10

def get_results_path(dim, n_terms):
    if args.test:
        return RESULTS_FOLDER + f'shapley_accuracy-n{dim}-T{n_terms}-test.csv'
    else:
        return RESULTS_FOLDER + f'shapley_accuracy-n{dim}-T{n_terms}.csv'
    
def get_fig_path(dim, n_terms):
    if args.test:
        return FIGURES_FOLDER + f'shapley_accuracy-n{dim}-T{n_terms}-test.pdf'
    else:
        return FIGURES_FOLDER + f'shapley_accuracy-n{dim}-T{n_terms}.pdf'

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

def SHAPIQ_time(model, X: np.ndarray, budget, random_state=0):
    start = time()
    approximator = shapiq.approximator.SHAPIQ(n=X.shape[1], index="SV", max_order=1)
    explainer = shapiq.TabularExplainer(
        model=model,
        data=X,
        imputer="marginal",
        approximator=approximator,
        index="SV",  
        max_order=1, 
        sample_size=X.shape[0],
        random_state=random_state,
    )
    shapiq_values = explainer.explain_X(X, budget=budget)
    return time()-start, np.array([values.get_n_order_values(1) for values in shapiq_values]) 

def run_expe():
    for i in range(len(DIMS)):
        DIM = DIMS[i]
        MAX_FS_LENGTH = MAX_FS_LENGTHS[i]
        BUDGET = BUDGETS[i]
        N_TERM = N_TERMS[i]

        total_time_start = time()
        df = pd.DataFrame()
        X = generate_boolean_dataset(N_SAMPLES, n_dim=DIM, rng=0)
        final_k_for_star_shap = False
        for k in range(MIN_FS_LENGTH, MAX_FS_LENGTH+1):
            if not final_k_for_star_shap:
                model = SOUM_Generator(DIM, N_TERM, min_fs_length=k, max_fs_length=k).sample()
                star_time, star_values = STAR_time(model, X)
                star_results = pd.DataFrame({'Algorithm' : [f'{k}-STAR-SHAP'], 
                                            'n' : [DIM], 
                                            'time' : [star_time], 
                                            'max fs size' : [k],
                                            'mean MSE' : [0]})
                if VERBOSE:
                    print(f'STAR time for FS size {k} and n={DIM}  : {round(star_time, 2)} seconds.')
                df = pd.concat([df, star_results], ignore_index=True)
            for budget in BUDGET:
                skip_budget = False
                total_shapiq_time = 0
                for n_run in range(N_SHAPIQ_RUNS):
                    if not skip_budget:
                        shapiq_time, shapiq_values = SHAPIQ_time(model, X, budget, random_state=n_run)
                        total_shapiq_time += shapiq_time
                        r2 = r2_score(y_true=np.array(star_values).flatten(), y_pred=np.array(shapiq_values).flatten())
                        if n_run == 0:
                            if r2 <= -0.25:
                                skip_budget = True
                        if not skip_budget:
                            shap_results = pd.DataFrame({'Algorithm' : ['SHAP-IQ'], 'n' : [DIM], 
                                                        'time' : [shapiq_time], 
                                                        'budget' : [budget],
                                                        'max fs size' : [k], 
                                                        'r2' : [r2]})
                            df = pd.concat([df, shap_results], ignore_index=True)
                if VERBOSE:
                    print(f'SHAP-IQ budget={budget} total time : {round(total_shapiq_time, 2)} seconds.')
        df.to_csv(get_results_path(DIM, N_TERM))
        print(f'Total experiment time : {round(time() - total_time_start, 0)} seconds.')

def make_figs():
    for j in range(len(DIMS)):
        df = pd.read_csv(get_results_path(DIMS[j], N_TERMS[j]), index_col=0)
        max_time = max(df['time'])

        # Group data and prepare for plotting
        grouped = df.groupby(['Algorithm', 'max fs size'])
        ratio = 4/3
        edge = 3.5
        fig, ax = plt.subplots(figsize=(ratio * edge, edge), dpi=300)
        plt.grid(axis='y', which='both', linestyle='-', linewidth=0.5, color='lightgray', zorder=1)

        MARKERS = ['+', 'x', '*', 'v', '1', 'D', 'o']
        y_max = 0.6
        marker_i = 0
        # Loop over groups for plotting
        for i, ((algo, k), group) in enumerate(grouped):
            # Separate STAR-SHAP and SHAP-IQ for different plotting requirements
            if 'STAR-SHAP' in algo:
                # Vertical line to indicate STAR-SHAP time
                x_val = max(group['time'])
                plt.axvline(x=x_val + 0.008*max_time, color="gray", linestyle=":", linewidth=1)
                plt.text(x=x_val, y=y_max, s=algo, rotation=90, ha='center', va='bottom', 
                        color="gray", fontsize=10, fontweight='bold')
            else:  # SHAP-IQ
                agg_group = group.groupby('budget').agg({
                    'r2': 'mean',
                    'time': 'mean',
                }).reset_index().sort_values('time')
                plt.plot(
                    agg_group['time'], 
                    1-agg_group['r2'],
                    label=f'{k}-SHAP-IQ',
                    linewidth=0.8, antialiased=True, zorder=3,
                    marker=MARKERS[marker_i % len(MARKERS)], markersize=4
                )
                marker_i += 1

        # Finalize plot
        ax.set_axisbelow(False)
        ax.set_xlabel('Time (s)', fontsize=14)
        plt.ylabel('1 - Coefficient of determination', fontsize=14, labelpad=10)
        plt.legend(frameon=True, fontsize=9, facecolor='white', framealpha=1, loc='upper right')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(get_fig_path(DIMS[j], N_TERMS[j]), bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    if RUN:
        run_expe()
    make_figs()
    