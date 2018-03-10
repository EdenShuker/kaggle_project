import inspect
import os
import sys

code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)
import xgboost as xgb
import numpy as np


class XGBC(object):
    def __init__(self, num_round=2, max_depth=2, eta=1.0, min_child_weight=2, colsample_bytree=1,
                 objective='multi:softprob'):
        self.max_depth = max_depth
        self.eta = eta
        self.colsample_bytree = colsample_bytree
        self.num_round = num_round
        self.min_child_weight = min_child_weight
        self.objective = objective

    def fit(self, train, label):
        dtrain = xgb.DMatrix(train, label=label, missing=-999)
        param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': 1,
                 'colsample_bytree': self.colsample_bytree, 'min_child_weight': self.min_child_weight,
                 'objective': self.objective,
                 'num_class': 9}
        self.bst = xgb.train(param, dtrain, self.num_round)

    def predict_proba(self, test):
        dtest = xgb.DMatrix(test, missing=-999)
        ypred = self.bst.predict(dtest)
        return ypred
