import argparse

from interpret.glassbox import ExplainableBoostingRegressor as EBM
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from typing import Union

from dataset_loaders import *
from expe_utils import TABLES_FOLDER
from expe_utils import RegressionExperiment, TableMaker
from FeatureSubsets import FSModel
from Learners import *
from Model import *

def parse():
    parser = argparse.ArgumentParser(description='Regression experiment for STAR RKHS Weightings of functions.')
    parser.add_argument('-t', '--test', action='store_true', help='Run test version of the experiment.')
    parser.add_argument('--info', action='store_true', help='Print information on the datasets.')
    parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the tables.')
    parser.add_argument('--params', action='store_true', help='Print the best params.')
    parser.add_argument('--final', action='store_true', help='Use the true test set. Only meant to be used once!')
    return parser.parse_args()


RNG = np.random.default_rng(0)
pd.set_option("display.precision", 3)

LEARNERS = [LeastSquaresLearner]
TABLE_COLUMNS = ['algorithm', 'dataset', 'train R2', 'test R2', 'fit time']
TABLE_COLUMN_NAMES = ['Training $R^2$', 'Test $R^2$', 'Training time (s)']
TABLE_DATASETS = ['abalone',
                  'diabetes', 
                  'housing', 
                  'concrete', 
                  'conductivity', 
                  'wine']

def _get_loaders(final: bool, train_size):
    kwargs = {'final' : final, 'scale_x' : True, 'scale_y' : True, 'train_size' : train_size}
    return [
        DiabetesLoader(**kwargs),
        CaliforniaHousingLoader(**kwargs),
        ConcreteLoader(**kwargs),
        WineLoader(**kwargs),
        ConductivityLoader(**kwargs),
        AbaloneLoader(**kwargs),
    ]

def get_test_loaders():
    return _get_loaders(False, 0.20)

def get_almost_final_loaders():
   return _get_loaders(False, 0.75)

def get_final_loaders():
   return _get_loaders(True, 0.75)

def bold_max(s: pd.Series):
    """
    Bold the maximum value in a series.
    """
    try:
        is_close_to_max = (s.max() - s <= 0.002)
        return ['\\textbf{%.3f}' % v if m else '%.3f' % v for v, m in zip(s, is_close_to_max)]
    except:
        return s

def apply_bold_max(group, columns):
    for col in columns:
        group[col] = bold_max(group[col])
    return group

def print_best_params(experiment: RegressionExperiment):
    df = pd.read_csv(experiment.get_results_path(), index_col=0)
    print(df[['dataset', 'algorithm', 'max_theta', 'regularization']])


class TestParams:
    def __init__(self) -> None:
        self.N_RUNS = 1
        self.LEARNER_PARAMS = {'n_iter' : [1],
                               'regularization' : [0.00001],
                               'batch_size' : [1]}
        self.I1_PARAMS = {'max_theta' : [0.9]}
        self.I2_PARAMS = {'max_theta' : [0.9]}
        self.I3_PARAMS = {'sigma' : [1],
                          'gamma' : [1]}
        self.FS_LENGTHS = [1, [1, 2]]
        self.FS_PARAMS = {'fs_length' : [1], 'fs_type' : ['any']}
        self.KR_PARAMS = {'kernel' : ['rbf'], 'alpha' : [1]}
        self.SVM_PARAMS = {'C' : [1]}
        self.TREE_PARAMS = {'max_depth' : [2]}
        self.EBM_PARAMS = {'max_bins': [512],
                           'learning_rate': [0.005],
                           'max_rounds': [1000],
                           'min_samples_leaf': [1],
                           'random_state': [0]}


class AlmostFinalParams:
    def __init__(self) -> None:    
        self.N_RUNS = 1
        self.LEARNER_PARAMS = {'n_iter' : [1000],
                               'regularization' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                               'batch_size' : [50]}
        self.I1_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.I2_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.I3_PARAMS = {'sigma' : [0.01, 0.1, 1],
                          'gamma' : [0.01, 0.1, 1]}
        self.FS_LENGTHS = [1, 2, 3, 4, 5, [1, 2, 3, 4, 5]]
        self.FS_PARAMS = {'fs_type' : ['any']}
        self.KR_PARAMS = {'kernel' : ['rbf'], 'alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]}
        self.SVM_PARAMS = {'C' : [0.5, 1, 5, 10, 50]}
        self.TREE_PARAMS = {'max_depth' : [2, 5, 10]}
        self.EBM_PARAMS = {'max_bins': [512, 1024, 2048],
                           'learning_rate': [0.005, 0.01, 0.02],
                           'max_rounds': [15000, 25000, 35000],
                           'min_samples_leaf': [1, 2, 3],
                           'random_state': [0]}


class FinalParams:
    def __init__(self) -> None:    
        self.N_RUNS = 1
        self.LEARNER_PARAMS = {'n_iter' : [5000],
                        'regularization' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'batch_size' : [50]}
        self.I1_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.I2_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.I3_PARAMS = {'sigma' : [0.01, 0.1, 1],
                          'gamma' : [0.01, 0.1, 1]}
        self.FS_LENGTHS = [1, 2, 3, 4, 5, [1, 2, 3, 4, 5]]
        self.FS_PARAMS = {'fs_type' : ['any']}
        self.KR_PARAMS = {'kernel' : ['rbf'], 'alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]}
        self.SVM_PARAMS = {'C' : [0.5, 1, 5, 10, 50]}
        self.TREE_PARAMS = {'max_depth' : [2, 5, 10, 20]}
        self.EBM_PARAMS = {'max_bins': [512, 1024, 2048],
                           'learning_rate': [0.005, 0.01, 0.02],
                           'max_rounds': [15000, 25000, 35000],
                           'min_samples_leaf': [1, 2, 3],
                           'random_state': [0]}


class JMLR2024RegressionExperiment(RegressionExperiment):
    def get_models(self, params: Union[TestParams, AlmostFinalParams, FinalParams]):
        FS_I1_PARAMS = {**params.FS_PARAMS, **{'base_model_params' : list(ParameterGrid(params.I1_PARAMS))}}
        FS_I1_PARAMS.update({'base_model_class' : [RWSign]})
        FS_I2_PARAMS = {**params.FS_PARAMS, **{'base_model_params' : list(ParameterGrid(params.I2_PARAMS))}}
        FS_I2_PARAMS.update({'base_model_class' : [RWRelu]})    
        MODELS = []
        MODELS.append((RWSign, params.I1_PARAMS, 'RWSign'))
        MODELS.append((RWRelu, params.I2_PARAMS, 'RWRelu'))
        MODELS.append((RWStumps, params.I3_PARAMS, 'RWStumps'))
        for k in params.FS_LENGTHS:
            MODELS.append((FSModel, {**FS_I1_PARAMS, 'fs_length' : [k]}, f'RWSign {k}-STAR'))
            MODELS.append((FSModel, {**FS_I2_PARAMS, 'fs_length' : [k]}, f'RWRelu {k}-STAR'))
        return MODELS

    def get_sklearn_algos(self, params: Union[TestParams, AlmostFinalParams, FinalParams]):
        SKLEARN_ALGOS = []
        SKLEARN_ALGOS.append((KernelRidge(), params.KR_PARAMS))  
        SKLEARN_ALGOS.append((SVR(), params.SVM_PARAMS))  
        SKLEARN_ALGOS.append((DecisionTreeRegressor(), params.TREE_PARAMS))   
        SKLEARN_ALGOS.append((LinearRegression(), {}))
        SKLEARN_ALGOS.append((EBM(), params.EBM_PARAMS))
        return SKLEARN_ALGOS


class TestExperiment(JMLR2024RegressionExperiment):
    def __init__(self) -> None:
        super().__init__('regression-test', get_test_loaders(), TestParams(), LEARNERS=LEARNERS)

class AlmostFinalExperiment(JMLR2024RegressionExperiment):
    def __init__(self) -> None:
        super().__init__('regression', get_almost_final_loaders(), AlmostFinalParams(), LEARNERS=LEARNERS)

class FinalExperiment(JMLR2024RegressionExperiment):
    def __init__(self) -> None:
        super().__init__('regression-final', get_final_loaders(), FinalParams(), LEARNERS=LEARNERS)


class AllInfoTable(TableMaker):
    def __init__(self, experiment: RegressionExperiment, TABLE_COLUMNS, TABLE_COLUMN_NAMES, TABLE_DATASETS) -> None:
        self.TABLE_COLUMNS = TABLE_COLUMNS
        self.TABLE_COLUMN_NAMES = TABLE_COLUMN_NAMES
        self.TABLE_DATASETS = TABLE_DATASETS
        super().__init__(experiment)

    def _get_table_from_df(self, df: pd.DataFrame, function='mean'):
        table = df[self.TABLE_COLUMNS]
        mask = table['dataset'].isin(self.TABLE_DATASETS)
        table = table[mask]
        if function == 'mean':
            table = table.groupby(['dataset', 'algorithm']).mean()
        elif function == 'std':
            table = table.groupby(['dataset', 'algorithm']).std()
        return table
    
    def save_tables(self):
        self.mean_table.to_latex(self.get_mean_table_path(), header=self.TABLE_COLUMN_NAMES, escape=False, float_format="%.3f")
        self.table_with_std.to_latex(self.get_table_with_std_path(), header=self.TABLE_COLUMN_NAMES, escape=False, float_format="%.3f")

    def get_mean_table_path(self) -> str:
        return TABLES_FOLDER + self.experiment.name + '.tex'
    
    def get_table_with_std_path(self) -> str:
        return TABLES_FOLDER + self.experiment.name + '-with-std.tex'
    

class DenseRegressionTable(TableMaker):
    def __init__(self, experiment: RegressionExperiment, TABLE_COLUMNS, TABLE_DATASETS) -> None:
        self.TABLE_COLUMNS = TABLE_COLUMNS
        self.TABLE_DATASETS = TABLE_DATASETS
        super().__init__(experiment)

    def _get_table_from_df(self, df: pd.DataFrame, function='mean'):
        if function == 'mean':
            grouped_df = df[self.TABLE_COLUMNS].groupby(['dataset', 'algorithm']).mean().reset_index()
        elif function == 'std':
            grouped_df = df[self.TABLE_COLUMNS].groupby(['dataset', 'algorithm']).std().reset_index()
        pivot_df = grouped_df.pivot(index='algorithm', columns='dataset', values='test R2')
        pivot_df = pivot_df.apply(bold_max)
        return pivot_df

    def save_tables(self):
        self.mean_table[self.TABLE_DATASETS].to_latex(self.get_mean_table_path(), escape=False, float_format="%.3f")
        self.table_with_std[self.TABLE_DATASETS].to_latex(self.get_table_with_std_path(), escape=False, float_format="%.3f")

    def get_mean_table_path(self) -> str:
        return TABLES_FOLDER + self.experiment.name + '-dense.tex'
    
    def get_table_with_std_path(self) -> str:
        return TABLES_FOLDER + self.experiment.name + '-dense-with-std.tex'

if __name__ == '__main__':
    args = parse()
    if args.test:
        experiment = TestExperiment()
    elif args.final:
        experiment = FinalExperiment()
    else:
        experiment = AlmostFinalExperiment()
    
    if args.info:
        for loader in experiment.DATASET_LOADERS:
            loader.info()
            print('-------------------------------------------')

    if not args.norun:
        experiment.launch()

    normal_table_maker = AllInfoTable(experiment, TABLE_COLUMNS, TABLE_COLUMN_NAMES, TABLE_DATASETS)
    dense_table_maker = DenseRegressionTable(experiment, TABLE_COLUMNS, TABLE_DATASETS)
    normal_table_maker.generate_and_save_tables()
    dense_table_maker.generate_and_save_tables()

    if args.params:
        print_best_params(experiment)
