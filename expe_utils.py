import pandas as pd
import pickle
from sklearn.metrics import r2_score
from pygam import LinearGAM
from sklearn.model_selection import GridSearchCV

from time import time
from utils import ensure_folder_exists

from dataset_loaders import Loader
from Learners import *
from Model import *
from Pruners import *
from RKHSWeightings import RKHSWeightingCV

RESULTS_FOLDER = './results/' ; ensure_folder_exists(RESULTS_FOLDER)
TABLES_FOLDER = './tables/' ; ensure_folder_exists(TABLES_FOLDER)

def get_R2(clf, data_loader: Loader):
    results = {}
    X_train, y_train = data_loader.get_train_X_y()
    X_test, y_test = data_loader.get_test_X_y()
    start_pred = time()
    train_pred = clf.predict(X_train)
    results['train pred time'] = [time() - start_pred]
    test_pred = clf.predict(X_test)
    results['train R2'] = [r2_score(y_train, train_pred)]
    results['test R2'] = [r2_score(y_test, test_pred)]
    return results

def clear_path(path):
    try:
        os.remove(path)
    except Exception as e:
        pass

class FittedModel:
    def __init__(self, model, model_name, dataset_loader):
        self.model = model
        self.model_name = model_name
        self.dataset_loader = dataset_loader

class TimeTracker:
    def __init__(self, n_fits):
        self.start = time()
        self.n_completed_fits = 0
        self.n_fits = n_fits

    def update(self):
        self.n_completed_fits += 1
        time_so_far = time() - self.start
        time_remaining = time_so_far / self.n_completed_fits * (self.n_fits - self.n_completed_fits)
        time_so_far = round(time_so_far/3600, 2)
        time_remaining = round(time_remaining/3600, 2)
        print("Fit {} of {} done. Elapsed time : {} hours.".format(self.n_completed_fits, self.n_fits, time_so_far))
        print("Estimated time remaining : {} hours.".format(time_remaining))

class RegressionExperiment:
    def __init__(self, name, DATASET_LOADERS: List[Loader], PARAMS, LEARNERS, RNG=0) -> None:
        self.name = name
        self.PARAMS = PARAMS
        self.LEARNERS = LEARNERS
        self.RNG = np.random.default_rng(RNG)
        self.DATASET_LOADERS = DATASET_LOADERS
        self.MODELS = self.get_models(PARAMS)
        self.SKLEARN_ALGOS = self.get_sklearn_algos(PARAMS)
        self.time_tracker = TimeTracker(self.get_n_total_fits())

    def get_models(self):
        raise NotImplementedError
    
    def get_sklearn_algos(self):
        raise NotImplementedError

    def get_n_total_fits(self):
        N_SKLEARN = len(self.SKLEARN_ALGOS)
        N_RKHS = len(self.MODELS)
        return (N_SKLEARN + self.PARAMS.N_RUNS * N_RKHS) * len(self.DATASET_LOADERS)

    def run_rkhs_weighting(self):
        """Run experiments for SFGD instantiations.

        N_RUNS experiments per combination of: 
        dataset, learner, model.
         
        One experiment means crossvalidating to find the best hyperparameters
        then calculating various metrics on the
        final model. Results are saved in the results folder.
        """
        for dataset in self.DATASET_LOADERS:
            for learner_class in self.LEARNERS:
                for (model_class, params_grid, name) in self.MODELS:
                    print(f'Now fitting {dataset.name} with {name}')
                    for _ in range(self.PARAMS.N_RUNS): 
                        self.run_one_rkhs_weighting(dataset, learner_class, model_class, params_grid, name)
                        self.time_tracker.update()

    def run_one_rkhs_weighting(self, data_loader: Loader, learner_class, model_class, params_grid, name):
        results = {'algorithm' : name}
        results['dataset'] = [data_loader.name]
        cv = RKHSWeightingCV(RKHSWeightingRegressor, learner_class, model_class, 
                         self.PARAMS.LEARNER_PARAMS, params_grid, rng=self.RNG, verbose=False)
        X, y = data_loader.get_train_X_y()
        cv.fit(X, y)
        clf = cv.best_estimator_
        results['fit time'] = cv.refit_time_
        for key in cv.best_learner_params_:
            results[key] = str(cv.best_learner_params_[key])
        for key in cv.best_model_params_:
            results[key] = str(cv.best_model_params_[key])
        results['rademacher'] = [clf.learner.rademacher_bound()]
        results['norm'] = [clf.model.norm()]
        results.update(get_R2(cv.best_estimator_, data_loader))
        self.write_results(results)

    def run_sklearn(self):
        """Run experiments for sklearn compatible algorithms.

        Simple crossvalidation with training and test errors.
        Saves results in the results folder.
        """
        for dataset in self.DATASET_LOADERS:
            for clf, params in self.SKLEARN_ALGOS:
                print(f'Now fitting {dataset.name} with {type(clf).__name__}')
                self.run_one_sklearn(dataset, clf, params)
                self.time_tracker.update()

    def run_one_sklearn(self, data_loader: Loader, clf, params):
        algo_name = type(clf).__name__
        results = {'algorithm' : algo_name}
        results['dataset'] = [data_loader.name]
        cv = GridSearchCV(clf, params, cv=5, verbose=0, n_jobs=-1)
        X, y = data_loader.get_train_X_y()
        cv.fit(X, y)
        results['fit time'] = cv.refit_time_
        for key in cv.best_params_:
            results[key] = cv.best_params_[key]
        results.update(get_R2(cv.best_estimator_, data_loader))
        self.write_results(results)

    def run_gam(self):
        for data_loader in self.DATASET_LOADERS:
            algo_name = 'Splines GAM'
            results = {'algorithm' : algo_name}
            results['dataset'] = [data_loader.name]
            X, y = data_loader.get_train_X_y()
            start = time()
            lams = np.tile(self.GAM_LAMS, (X.shape[1], 1)).T
            gam = LinearGAM().gridsearch(X, y, lam=lams) 
            results['fit time'] = (time() - start) / len(self.GAM_LAMS)
            results.update(get_R2(gam, data_loader))
            self.write_results(results)
            self.time_tracker.update()

    def write_results(self, results: dict):
        df = pd.DataFrame.from_dict(results)
        path = self.get_results_path()
        if os.path.exists(path):
            previous_df = pd.read_csv(path, index_col=0)
            df = pd.concat((previous_df, df), sort=True)
            df.index = pd.Index(np.arange(len(df.index)))
        df.to_csv(path)

    def save_model(self, model, model_name, dataset_loader):
        fitted_model = FittedModel(model, model_name, dataset_loader)
        path = self.get_pickle_path()
        try:
            with open(path, "rb") as file:
                fitted_list = pickle.load(file)
        except Exception:
            fitted_list = []
        fitted_list.append(fitted_model)
        with open(path, "wb") as file:
            pickle.dump(fitted_list, file)

    def launch(self):
        clear_path(self.get_results_path())
        clear_path(self.get_pickle_path())
        self.run_rkhs_weighting()
        self.run_sklearn()
        print("results saved in " + self.get_results_path())

    def get_results_path(self) -> str:
        return RESULTS_FOLDER + self.name + '.csv'
    
    def get_pickle_path(self) -> str:
        return RESULTS_FOLDER + self.name + '.pkl'
        

class TableMaker:
    def __init__(self, experiment: RegressionExperiment) -> None:
        self.experiment = experiment

    def generate_and_save_tables(self):
        df = self.get_raw_df()
        df = self.clean_df(df)
        self.generate_tables(df)
        self.save_tables()
        print(f"Tables saved to {TABLES_FOLDER}")

    def get_raw_df(self):
        return pd.read_csv(self.experiment.get_results_path(), index_col=0)
    
    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.replace('ExplainableBoostingRegressor', 'EBM', regex=True)
        cleaned_df = cleaned_df.replace('california housing', 'housing', regex=True)
        cleaned_df = cleaned_df.replace(np.nan, '', regex=True)
        cleaned_df = cleaned_df.replace('nan', '', regex=True)
        return cleaned_df

    def generate_tables(self, df: pd.DataFrame):
        self.mean_table = self.get_mean_table_from_df(df)
        self.std_table = self.get_std_table_from_df(df)
        self.table_with_std = self.get_table_with_std(self.mean_table, self.std_table)

    def get_mean_table_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._get_table_from_df(df, 'mean')
    
    def get_std_table_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._get_table_from_df(df, 'std')
    
    def _get_table_from_df(self, df: pd.DataFrame, function='mean'):
        raise NotImplementedError

    def get_table_with_std(self, table: pd.DataFrame, std_table: pd.DataFrame):
        table_with_std = table.astype(str) + ' ± ' + std_table.astype(str)
        table_with_std = table_with_std.replace(' ± ', '', regex=False)
        return table_with_std
    
    def save_tables(self):
        raise NotImplementedError