import gzip
import os
import urllib

import idx2numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, add_dummy_feature
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from ucimlrepo import fetch_ucirepo 

DATA_FOLDER = './datasets/'

def scale_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def scale_targets(y_train, y_test):
    y_scaler = YScaler()
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    return y_train_scaled, y_test_scaled

def add_bias(X_train, X_test):
    X_train = add_dummy_feature(X_train)
    X_test = add_dummy_feature(X_test)
    return X_train, X_test

class YScaler:
    def fit(self, y):
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)

    def transform(self, y):
        return (y - self.mean_)/self.std_
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def restore(self, transformed_y):
        return transformed_y * self.std_ + self.mean_

def sample_batch(X: np.ndarray, y: np.ndarray, batch_size=100, rng=0):
    sample_size, n_features = X.shape
    rng = np.random.default_rng(rng)
    batch_idx = rng.choice(sample_size, size=batch_size, replace=True)
    batch_data = X[batch_idx, :].reshape((batch_size, n_features))
    batch_targets = y[batch_idx]
    return batch_data, batch_targets  


class Loader:
    def __init__(self, name, final=False, scale_x=True, scale_y=False, train_size=0.75, random_state=0) -> None:
        self.name = name
        self.final = final
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.train_size = train_size
        self.random_state = random_state
        X_train, X_test, y_train, y_test = self.load()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def load(self):
        if self.final:
            return self._load_full(self.train_size)
        else:
            return self._load_valid(self.train_size)
        
    def load_raw(self):
        raise NotImplementedError

    def _load_full(self, train_size=None):
        X, y = self.load_raw()
        train_size = train_size if train_size is not None else 0.75
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, random_state=self.random_state)
        if self.scale_x:
            X_train, X_test = scale_data(X_train, X_test)
        if self.scale_y:
            y_train, y_test = scale_targets(y_train, y_test)
        return X_train, X_test, y_train, y_test
    
    def _load_valid(self, train_size=None):
        X_train, _, y_train, _ = self._load_full()
        train_size = train_size if train_size is not None else 0.75
        return train_test_split(X_train, y_train, train_size=self.train_size, random_state=self.random_state)
    
    def get_train_X_y(self):
        return self.X_train, self.y_train
    
    def get_test_X_y(self):
        return self.X_test, self.y_test

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def info(self):
        print(f'Name of the dataset : {self.name}')
        print(f'Train size : {self.X_train.shape[0]} examples of {self.X_train.shape[1]} dimensions')
        print(f'Test size : {self.X_test.shape[0]} examples')

    def get_filename_X(self):
        return DATA_FOLDER + self.name + '_X.csv'

    def get_filename_y(self):
        return DATA_FOLDER + self.name + '_y.csv'

class UCI_Loader(Loader):
    def load_uci(self, id):
        try:
            X = np.loadtxt(self.get_filename_X())
            y = np.loadtxt(self.get_filename_y())
        except Exception as e:
            print(f'fetching {self.name} from uci repository...')
            dataset = fetch_ucirepo(id=id)
            X = dataset.data.features 
            y = dataset.data.targets 
            X = pd.get_dummies(X).to_numpy()
            y = np.ravel(y.to_numpy())
            np.savetxt(self.get_filename_X(), X)
            np.savetxt(self.get_filename_y(), y)
        return X, y
    
class UCR_Loader(Loader):
    def __init__(self, regression: bool, segment=-1, **kwargs) -> None:
        self.regression = regression
        self.segment = segment
        super().__init__(**kwargs)

    def load_ucr(self):
        train_name = DATA_FOLDER + 'UCRArchive_2018/' + self.name + '/' + self.name + '_TRAIN.tsv'
        test_name = DATA_FOLDER + 'UCRArchive_2018/' + self.name + '/' + self.name + '_TEST.tsv'

        train_data = pd.read_csv(train_name, sep='\t').interpolate().values
        test_data = pd.read_csv(test_name, sep='\t').interpolate().values

        X_train = train_data[:, 1:]
        y_train = train_data[:, 0]

        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        return X_train, X_test, y_train, y_test
    
    def _load_full(self):
        X_train, X_test, y_train, y_test = self.load_ucr()
        if self.regression:
            X_train, X_test, y_train, y_test = self.regressionify(X_train, X_test)            
        if self.scale_x:
            X_train, X_test = scale_data(X_train, X_test)
        if self.scale_y:
            y_train, y_test = scale_targets(y_train, y_test)
        return X_train, X_test, y_train, y_test

    def regressionify(self, X_train: np.ndarray, X_test: np.ndarray):
        if self.segment < 2:
            X_train = X_train[:, :-1]
            y_train = X_train[:, -1]
            X_test = X_test[:, :-1]
            y_test = X_test[:, -1]
        else:
            X_train, y_train = self.segmentify(X_train)
            X_test, y_test = self.segmentify(X_test)
        return X_train, X_test, y_train, y_test
    
    def segmentify(self, X: np.ndarray):
        if self.segment >= 2:
            m, n = X.shape
            n_full_segments = n // self.segment
            rest = n % self.segment
            partial_segment = rest > min(n_full_segments / 2, 2)
            n_total_segments = n_full_segments + partial_segment
            X_new = np.zeros((m * n_total_segments, self.segment-1))
            y_new = np.zeros(m * n_total_segments)
            for n in range(n_full_segments):
                segment_start = n * self.segment
                segment_end = (n+1) * self.segment - 1
                X_new[n*m:(n+1)*m, :] = X[:, segment_start:segment_end]
                y_new[n*m:(n+1)*m] = X[:, segment_end]
            if partial_segment:
                X_new[m * n_full_segments:, :] = X[:, -self.segment:-1]
                y_new[m * n_full_segments:] = X[:, -1]
            return X_new, y_new
        else:
            print(f'Cannot segment with self.segment < 2')
            return X

#####################################################################
######################## Classification #############################
#####################################################################
# TODO MNIST doesn't work with scaler
class MNISTLoader(Loader):
    """Loader for the MNIST dataset.

    Will download the data if necessary.
    """
    def __init__(self, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], **kwargs):
        if len(digits) < 10:
            for d in digits:
                self.name += str(d)
        self.digits = digits
        super().__init__(name='mnist', **kwargs)

    def maybe_download(self, file_names):
        WEB_PATH = 'http://yann.lecun.com/exdb/mnist/'
        GZ = '.gz'

        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        for name in file_names:
            if not os.path.exists(name):
                if not os.path.exists(name + GZ):
                    with urllib.request.urlopen(WEB_PATH + name + GZ) as response:
                        file_content = response.read()
                    with open(name + GZ, 'wb') as f:
                        f.write(file_content)
                with gzip.open(name + GZ, 'rb') as f:
                    file_content = f.read()
                with open(name, 'wb') as f:
                    f.write(file_content)
        os.chdir('../')

    def _load_full(self):
        train_data_name = 'train-images-idx3-ubyte'
        train_labels_name = 'train-labels-idx1-ubyte'
        test_data_name = 't10k-images-idx3-ubyte'
        test_labels_name = 't10k-labels-idx1-ubyte'
        file_names = [train_data_name, train_labels_name, test_data_name, test_labels_name]
        self.maybe_download(file_names)

        X_train = idx2numpy.convert_from_file(DATA_FOLDER+train_data_name).astype(float)
        y_train = idx2numpy.convert_from_file(DATA_FOLDER+train_labels_name).astype(int)
        X_test = idx2numpy.convert_from_file(DATA_FOLDER+test_data_name).astype(float)
        y_test = idx2numpy.convert_from_file(DATA_FOLDER+test_labels_name).astype(int)

        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        train_indices = np.full(y_train.shape, False)
        test_indices = np.full(y_test.shape, False)

        for d in self.digits:
            train_indices = np.where((y_train == d) | train_indices, True, False)
            test_indices = np.where((y_test == d) | test_indices, True, False)

        X_train = X_train[train_indices,:]
        X_test = X_test[test_indices,:]
        y_train = y_train[train_indices]
        y_test = y_test[test_indices]

        if self.scale_x:
            X_train, X_test = scale_data(X_train, X_test)

        return X_train, X_test, y_train, y_test


class AdultsLoader(Loader):
    """Loader for the Adults dataset.

    Will download the data if necessary.

    Preprocesses categorical variables using one-hot encoding.    
    """
    def __init__(self, **kwargs):
        super().__init__(name='adults', **kwargs)

    def maybe_download(self, train_name, test_name):
        TRAIN_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        TEST_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        if not os.path.exists(train_name):
            train_df = pd.read_csv(TRAIN_WEB_PATH, header=None)
            train_df.to_csv(train_name)
        if not os.path.exists(test_name):
            test_df = pd.read_csv(TEST_WEB_PATH, header=None, skiprows=1)
            test_df.to_csv(test_name)
        os.chdir('../')
    
    def _load_full(self):
        train_name = 'adults_train.csv'
        test_name = 'adults_test.csv'

        self.maybe_download(train_name, test_name)

        train_df = pd.read_csv(DATA_FOLDER+train_name, index_col=0)
        test_df = pd.read_csv(DATA_FOLDER+test_name, index_col=0)
        n_train = train_df.shape[0]
        full_df = pd.concat((train_df, test_df), axis=0)
        full_df.replace(' <=50K.', ' <=50K', inplace=True)
        full_df.replace(' >50K.', ' >50K', inplace=True)
        full_df = pd.get_dummies(full_df)
        full_df.drop(full_df.columns[len(full_df.columns)-1], axis=1, inplace=True)
        data = full_df.to_numpy()
        Xtrain = data[:n_train,:-1].astype('int')
        ytrain = data[:n_train, -1].astype('int')
        Xtest = data[n_train:,:-1]
        ytest = data[n_train:, -1]
        if self.scale_x:
            X_train, X_test = scale_data(X_train, X_test)
        return Xtrain, Xtest, ytrain, ytest
    
class PlanningRelaxLoader(Loader):
    """Loader for the PlanningRelax dataset.

    Will download the data if necessary.   
    """
    def __init__(self, **kwargs):
        self.FILE_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt"
        self.FILE_NAME = "plrx.txt"
        super().__init__(name='planning relax', **kwargs)

    def maybe_download(self):
        
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        if not os.path.exists(self.FILE_NAME):
            urllib.request.urlretrieve(self.FILE_WEB_PATH, self.FILE_NAME)
        os.chdir('../')
    
    def load_raw(self):

        self.maybe_download()

        df = pd.read_csv(DATA_FOLDER+self.FILE_NAME, sep='\t', header=None)
        df = df.drop(columns=13)
        data = df.to_numpy()
        X = data[:,:-1]
        y = data[:, -1]
        return X, y


class SkinSegmentationLoader(Loader):
    """Loader for the SkinSegmentation dataset.

    Will download the data if necessary.   
    """
    def __init__(self, **kwargs):
        self.FILE_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
        self.FILE_NAME = "Skin_NonSkin.txt"
        super().__init__(name='skin segmentation', **kwargs)

    def maybe_download(self):        
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        if not os.path.exists(self.FILE_NAME):
            urllib.request.urlretrieve(self.FILE_WEB_PATH, self.FILE_NAME)
        os.chdir('../')
    
    def load_raw(self):
        self.maybe_download()
        df = pd.read_csv(DATA_FOLDER+self.FILE_NAME, sep='\t', header=None)
        data = df.to_numpy()
        X = data[:,:-1]
        y = data[:, -1]
        return X, y


class BreastCancerLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(name='breast cancer', **kwargs)

    def load_raw(self):
        return load_breast_cancer(return_X_y=True)


#####################################################################
########################## Regression ###############################
#####################################################################

class HousePricesLoader(Loader):
    """
        Loader for the House Prices dataset.  
    """
    def __init__(self, drop_non_numerical=True, drop_na=True, **kwargs):
        self.drop_non_numerical = drop_non_numerical
        self.drop_na = drop_na
        super().__init__(name='house prices', **kwargs)
    
    def load_raw(self):
        train_df = pd.read_csv(DATA_FOLDER+"house_prices/train.csv", index_col=0)
        y_train = train_df['SalePrice'].to_numpy()
        train_df.drop(columns=['SalePrice'], inplace=True)

        test_df = pd.read_csv(DATA_FOLDER+"house_prices/test.csv", index_col=0)
        n_train = train_df.shape[0]
        full_df = pd.concat((train_df, test_df), axis=0)
        if self.drop_non_numerical:
            non_numeric_cols = full_df.select_dtypes(exclude=['number']).columns
            full_df = full_df.drop(columns=non_numeric_cols)
        else:
            full_df = pd.get_dummies(full_df)
            full_df.replace(False, -1, inplace=True)
            full_df.replace(True, 1, inplace=True)
        if self.drop_na:
            cols_with_missing_values = full_df.columns[full_df.isnull().any()]
            full_df = full_df.drop(columns=cols_with_missing_values)
        else:
            full_df.fillna(full_df.mean(), inplace=True)
        data = full_df.to_numpy()
        X_train = data[:n_train,:]
        X_test = data[n_train:,:]
        return X_train, y_train
    
class DiabetesLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(name='diabetes', **kwargs)

    def load_raw(self):
        return load_diabetes(return_X_y=True)
    
class CaliforniaHousingLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(name='california housing', **kwargs)

    def load_raw(self):
        return fetch_california_housing(return_X_y=True)
    
class ConcreteLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    """
    def __init__(self, **kwargs):
        super().__init__(name='concrete', **kwargs)
  
    def load_raw(self):
        return self.load_uci(165)
    
class WineLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/109/wine
    """
    def __init__(self, **kwargs):
        super().__init__(name='wine', **kwargs)
  
    def load_raw(self):  
        return self.load_uci(109)
    
class ForestFiresLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/162/forest+fires
    """
    def __init__(self, **kwargs):
        super().__init__(name='forest fires', **kwargs)
  
    def load_raw(self): 
        return self.load_uci(162)

class ConductivityLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/464/superconductivty+data
    """
    def __init__(self, **kwargs):
        super().__init__(name='conductivity', **kwargs)
  
    def load_raw(self): 
        return self.load_uci(464)
    
class AbaloneLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/1/abalone
    """
    def __init__(self, **kwargs):
        super().__init__(name='abalone', **kwargs)
  
    def load_raw(self): 
        return self.load_uci(1)
    
class LiverLoader(UCI_Loader):
    """
        https://archive.ics.uci.edu/dataset/60/liver+disorders
    """
    def __init__(self, **kwargs):
        super().__init__(name='liver', **kwargs)
  
    def load_raw(self): 
        return self.load_uci(60)
    

#####################################################################
########################## Time Series ##############################
#####################################################################

class ElectricDevicesLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='ElectricDevices', regression=regression, **kwargs)

class ChlorineConcentrationLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='ChlorineConcentration', regression=regression, **kwargs)

class ComputersLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='Computers', regression=regression, **kwargs)

class ECG5000Loader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='ECG5000', regression=regression, **kwargs)

class FacesUCRLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='FacesUCR', regression=regression, **kwargs)

class FordALoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='FordA', regression=regression, **kwargs)

class LargeKitchenAppliancesLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='LargeKitchenAppliances', regression=regression, **kwargs)

class NonInvasiveFetalECGThorax1Loader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='NonInvasiveFetalECGThorax1', regression=regression, **kwargs)

class StarLightCurvesLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='StarLightCurves', regression=regression, **kwargs)

class MelbournePedestrianLoader(UCR_Loader):
    def __init__(self, regression=False, **kwargs) -> None:
        super().__init__(name='MelbournePedestrian', regression=regression, **kwargs)