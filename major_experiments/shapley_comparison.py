import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap

from interpret.glassbox import ExplainableBoostingRegressor as EBM
from sklearn.datasets import fetch_california_housing

from dataset_loaders import CaliforniaHousingLoader
from expe_utils import RESULTS_FOLDER, FIGURES_FOLDER
from FeatureSubsets import FSModel
from Learners import *
from Model import *
from RKHSWeightings import RKHSWeightingRegressor
from Shapley import STAR_Explainer

parser = argparse.ArgumentParser(description='Comparison of the Shapley values of EBM versus STAR RKHS model.')
parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the figure.')
parser.add_argument('--test', action='store_true', help='Quicker experiment for testing purposes.')
args = parser.parse_args()

ratio = 4/3
edge = 5

RNG = np.random.default_rng(0)
RUN = not args.norun
if args.test:
    N_ITER = 50
    TRAIN_SIZE = 50
    BACKGROUND_SIZE = 50
    TEST_SIZE = 50
    RKHS_PICKLE = RESULTS_FOLDER + "./rkhs_explanation-test.pkl"
    EBM_PICKLE = RESULTS_FOLDER + "./ebm_explanation-test.pkl"
    RKHS_BEESWARM = FIGURES_FOLDER + "./beeswarm_plot_star_rkhs-test.pdf"
    EBM_BEESWARM = FIGURES_FOLDER + "./beeswarm_plot_ebm-test.pdf"
    RKHS_WATERFALL = FIGURES_FOLDER + "./waterfall_plot_star_rkhs-test.pdf"
    EBM_WATERFALL = FIGURES_FOLDER + "./waterfall_plot_ebm-test.pdf"
else:
    N_ITER = 5000
    TRAIN_SIZE = 0.75
    BACKGROUND_SIZE = 1000
    TEST_SIZE = 1000
    RKHS_PICKLE = RESULTS_FOLDER + "./rkhs_explanation.pkl"
    EBM_PICKLE = RESULTS_FOLDER + "./ebm_explanation.pkl"
    RKHS_BEESWARM = FIGURES_FOLDER + "./beeswarm_plot_star_rkhs.pdf"
    EBM_BEESWARM = FIGURES_FOLDER + "./beeswarm_plot_ebm.pdf"
    RKHS_WATERFALL = FIGURES_FOLDER + "./waterfall_plot_star_rkhs.pdf"
    EBM_WATERFALL = FIGURES_FOLDER + "./waterfall_plot_ebm.pdf"

feature_names = fetch_california_housing().feature_names
X_train, X_test, y_train, y_test = CaliforniaHousingLoader(final=True, train_size=TRAIN_SIZE, scale_x=True, scale_y=True).load()


if RUN:
    ebm = EBM(feature_names=feature_names, 
            learning_rate=0.01, 
            max_bins=512, 
            max_rounds=15000, 
            min_samples_leaf=2, 
            random_state=0)
    star_rkhs = FSModel(input=X_train,
                        base_model_class=RWRelu,
                        base_model_params={'max_theta' : 0.5},
                        fs_length=3,
                        fs_type='any',
                        rng=RNG)
    learner = LeastSquaresLearner(n_iter=N_ITER, regularization=1e-09, rng=RNG)
    clf = RKHSWeightingRegressor(learner=learner, model=star_rkhs)
    print('Training EBM...')
    ebm.fit(X_train, y_train)
    print('Training RKHS Weighting...')
    clf.fit(X_train, y_train)

    print('Calculating RKHS Weighting Shapley values...')
    star_explainer = STAR_Explainer(clf.model, X_train[:BACKGROUND_SIZE])
    rkhs_shap_values = star_explainer.shap_values(X_test[:TEST_SIZE], verbose=True)
    rkhs_explanation = shap.Explanation(
        values=rkhs_shap_values,
        data=X_train[:BACKGROUND_SIZE],           
        feature_names=feature_names
    )
    with open(RKHS_PICKLE, "wb") as f:
        pickle.dump(rkhs_explanation, f)

    print('Calculating EBM Shapley values...')
    tree_explainer = shap.ExactExplainer(ebm.predict, X_train[:BACKGROUND_SIZE])
    ebm_shap_values = tree_explainer(X_test[:TEST_SIZE])
    ebm_explanation = shap.Explanation(
        values=ebm_shap_values,
        data=X_train[:BACKGROUND_SIZE],       
        feature_names=feature_names
    )
    with open(EBM_PICKLE, "wb") as f:
        pickle.dump(ebm_explanation, f)
    print('Experiment done.')

with open(RKHS_PICKLE, "rb") as f:
    rkhs_explanation = pickle.load(f)

with open(EBM_PICKLE, "rb") as f:
    ebm_explanation = pickle.load(f)

ebm_explanation.data = X_train[:BACKGROUND_SIZE]
rkhs_explanation.data = X_train[:BACKGROUND_SIZE]

fig, ax = plt.subplots(figsize=(5, 5))
shap.plots.beeswarm(ebm_explanation, show=False)
fig.set_size_inches(5, 5)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
plt.tight_layout()
plt.savefig(EBM_BEESWARM, format="pdf")
plt.show() 
# plt.clf()



fig, ax = plt.subplots(figsize=(5, 5)) 
shap.plots.beeswarm(rkhs_explanation, show=False)
fig.set_size_inches(5, 5)
fig.patch.set_facecolor("white") 
ax.set_facecolor("white")
plt.tight_layout() 
plt.savefig(RKHS_BEESWARM, format="pdf")
plt.show()
# plt.clf()

example_index = 0
print(f'True y of the explained example : {y_test[0]}')

shap_values_single = ebm_explanation.values[example_index] 
data_single = ebm_explanation.data[example_index]
feature_names = ebm_explanation.feature_names
fig, ax = plt.subplots(figsize=(5, 5)) 
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_single,
        base_values=0,
        data=data_single,
        feature_names=feature_names
    ), show=False
)
fig.set_size_inches(ratio * edge, edge)
fig.patch.set_facecolor("white") 
ax.set_facecolor("white")
ax.axvline(x=0, color='black', linewidth=2, zorder=100)
plt.tight_layout() 
# plt.gca().grid(True)  # Get the current axis and turn off the grid
plt.savefig(EBM_WATERFALL, format="pdf")
plt.show()
# plt.clf()

shap_values_single = np.array(rkhs_explanation.values[example_index])
fig, ax = plt.subplots(figsize=(5, 5)) 
shap.plots.waterfall(
    shap.Explanation(
        values=np.array(shap_values_single),
        base_values=0,
        data=data_single,
        feature_names=feature_names
    ), show=False
)
fig.set_size_inches(ratio * edge, edge)
fig.patch.set_facecolor("white") 
ax.set_facecolor("white")
ax.axvline(x=0, color='black', linewidth=2, zorder=100)
plt.tight_layout() 
# plt.gca().grid(True)  # Get the current axis and turn off the grid
plt.savefig(RKHS_WATERFALL, format="pdf")
plt.show()
# plt.clf()