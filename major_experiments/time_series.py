import argparse
from interpret.glassbox import ExplainableBoostingRegressor as EBM
from sklearn.model_selection import ParameterGrid
from typing import Union

from dataset_loaders import *
from FeatureSubsets import FSModel
from Learners import LeastSquaresLearner
from Model import RWRelu
from regression import RegressionExperiment, AllInfoTable, DenseRegressionTable

def parse():
    parser = argparse.ArgumentParser(description='Time series prediction experiment for STAR RKHS Weightings of functions.')
    parser.add_argument('-t', '--test', action='store_true', help='Run test version of the experiment.')
    parser.add_argument('--info', action='store_true', help='Print information on the datasets.')
    parser.add_argument('--norun', action='store_true', help='Do not run the experiment. Only generate the tables.')
    parser.add_argument('--final', action='store_true', help='Use the true test set. Only meant to be used once!')
    return parser.parse_args()

LEARNERS = [LeastSquaresLearner]
TABLE_COLUMNS = ['algorithm', 'dataset', 'train R2', 'test R2', 'fit time']
TABLE_COLUMN_NAMES = ['Training $R^2$', 'Test $R^2$', 'Training time (s)']
TABLE_DATASETS = [
                  'ChlorineConcentration',
                  'Computers', 
                  'ECG5000', 
                  'FacesUCR',
                  'LargeKitchenAppliances',
                  'MelbournePedestrian'
                  ]

def _get_loaders(final: bool, train_size):
    kwargs = {'final' : final, 'regression' : True, 'scale_x' : True, 'scale_y' : True, 'train_size' : train_size, 'segment' : 11}
    return [ChlorineConcentrationLoader(**kwargs),
            ComputersLoader(**kwargs),
            ECG5000Loader(**kwargs),
            FacesUCRLoader(**kwargs),
            LargeKitchenAppliancesLoader(**kwargs),
            MelbournePedestrianLoader(**kwargs),
    ]

def get_test_loaders():
    return _get_loaders(False, 0.20)

def get_almost_final_loaders():
   return _get_loaders(False, 0.75)

def get_final_loaders():
   return _get_loaders(True, 0.75)


class TestParams:
    def __init__(self) -> None:
        self.N_RUNS = 1
        self.LEARNER_PARAMS = {'n_iter' : [1],
                        'regularization' : [0.00001],
                        'batch_size' : [1]}
        self.I2_PARAMS = {'max_theta' : [0.9]}
        self.FS_LENGTHS = [1, 2, 3, 4, 5]
        self.FS_PARAMS = {'fs_type' : ['seq']}
        self.EBM_PARAMS = {'max_bins': [512],
                           'learning_rate': [0.005],
                           'max_rounds': [15000],
                           'min_samples_leaf': [1]}


class AlmostFinalParams:
    def __init__(self) -> None:    
        self.N_RUNS = 1 
        self.LEARNER_PARAMS = {'n_iter' : [1000],
                        'regularization' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'batch_size' : [50]}
        self.I2_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.FS_LENGTHS = [1, 2, 3, 4, 5, [1, 2, 3, 4, 5]]
        self.FS_PARAMS = {'fs_type' : ['seq']}
        self.EBM_PARAMS = {'max_bins': [512, 1024, 2048],
                           'learning_rate': [0.005, 0.01, 0.02],
                           'max_rounds': [15000, 25000, 35000],
                           'min_samples_leaf': [1, 2, 3]}


class FinalParams:
    def __init__(self) -> None:    
        self.N_RUNS = 1 
        self.LEARNER_PARAMS = {'n_iter' : [5000],
                        'regularization' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                        'batch_size' : [50]}
        self.I2_PARAMS = {'max_theta' : [0.1, 0.5, 0.9]}
        self.FS_LENGTHS = [1, 2, 3, 4, 5, [1, 2, 3, 4, 5]]
        self.FS_PARAMS = {'fs_type' : ['seq']}
        self.EBM_PARAMS = {'max_bins': [512, 1024, 2048],
                           'learning_rate': [0.005, 0.01, 0.02],
                           'max_rounds': [15000, 25000, 35000],
                           'min_samples_leaf': [1, 2, 3]}


class TimeSeriesExperiment(RegressionExperiment):
    def get_models(self, params: Union[TestParams, AlmostFinalParams, FinalParams]):
        FS_I2_PARAMS = {**params.FS_PARAMS, **{'base_model_params' : list(ParameterGrid(params.I2_PARAMS))}}
        FS_I2_PARAMS.update({'base_model_class' : [RWRelu]})    
        MODELS = []
        MODELS.append((RWRelu, params.I2_PARAMS, 'RWRelu'))
        for k in params.FS_LENGTHS:
            MODELS.append((FSModel, {**FS_I2_PARAMS, 'fs_length' : [k]}, f'RWRelu {k}-STAR'))
        return MODELS

    def get_sklearn_algos(self, params: Union[TestParams, AlmostFinalParams, FinalParams]):
        SKLEARN_ALGOS = []
        SKLEARN_ALGOS.append((EBM(), params.EBM_PARAMS))
        return SKLEARN_ALGOS


class TimeSeriesTestExperiment(TimeSeriesExperiment):
    def __init__(self) -> None:
        super().__init__('time-series-test', get_test_loaders(), TestParams(), LEARNERS=LEARNERS)


class TimeSeriesAlmostFinalExperiment(TimeSeriesExperiment):
    def __init__(self) -> None:
        super().__init__('time-series', get_almost_final_loaders(), AlmostFinalParams(), LEARNERS=LEARNERS)


class TimeSeriesFinalExperiment(TimeSeriesExperiment):
    def __init__(self) -> None:
        super().__init__('time-series-final', get_final_loaders(), FinalParams(), LEARNERS=LEARNERS)

if __name__ == '__main__':
    args = parse()
    if args.test:
        experiment = TimeSeriesTestExperiment()
    elif args.final:
        experiment = TimeSeriesFinalExperiment()
    else:
        experiment = TimeSeriesAlmostFinalExperiment()
    
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
