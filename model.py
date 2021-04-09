import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    IsolationForest,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (
    SGDRegressor
)
from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from controls import *
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor



def get_acc(y_pred, y_true):
    correct = 0
    pos = 0
    samp = len(y_true)
    for i in range(len(y_true)):
        p, r = y_pred[i], y_true[i]
        print(p, r)
        if p * r > 0:
            correct += 1
        if r > 0:
            pos += 1
    return correct/samp, pos/samp 

class RegFeatures(object):

    def __init__(self):

        self.regs = [
            AdaBoostRegressor(),
            RandomForestRegressor(),
            SymbolicRegressor(),
            GradientBoostingRegressor(),
            BaggingRegressor(),
            KNeighborsRegressor(),
        ]
        self.names = [
            'AdaBoost',
            'RandomForest',
            'Symbolic',
            'GradientBoosting',
            'Bagging',
            'KNeighbors',
        ]

    def fit(self, X, y):
        for reg in self.regs:
            reg.fit(X, y)

    def load_features(self, X):
        rp = []
        for reg in self.regs:
            p = reg.predict(X)
            rp.append(p)
        return np.array(rp).T

    def check_acc(self, X, y):
        for i in range(len(self.regs)):
            reg = self.regs[i]
            nm = self.names[i]
            y_pred = reg.predict(X)
            acc = get_acc(y_pred, y)
            print(f'{nm}: {acc}')


class MultiRegressor(object):

    base_est = [
        CatBoostRegressor(
            iterations=1000,
            verbose=0,
            thread_count=-1,
            learning_rate=.02,
            depth=2,
        ), 
    ]
    def __init__(self, estimators=base_est, samples=3, verbose=1):

        self.samples = samples
        self.estimators = []
        self.n_est = len(estimators) * self.samples
        for est in estimators:
            for i in range(self.samples):
                self.estimators.append(est)
        self.verbose = verbose

    def fit(self, X, y):
        n = 1
        for est in self.estimators:
            est.fit(X, y)
            if self.verbose == 1:
                print(f'Trained Regressor {n} [{n}/{self.n_est}]')
            n += 1

    def predict(self, X):
        preds = []
        for est in self.estimators:
            p = np.array(est.predict(X))
            if len(preds) == 0:
                preds = p
            else:
                preds += p
        return preds/self.n_est

def feature_importances(X, y):
    est = RandomForestRegressor(n_jobs=-1)
    est.fit(X, y)
    return est.feature_importances_