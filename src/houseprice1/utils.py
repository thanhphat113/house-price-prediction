import os
import sys
from src.houseprice1.exception import CustomException
from src.houseprice1.logger import logging
import pandas as pd

# from dotenv import load_dotenv

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import numpy as np


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        raise CustomException(ex, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as ex:
        raise CustomException(ex, sys)


class create_quality_weighted_area(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            overall_qual = X[:, 0]
            gr_liv_area = X[:, 1]

            weighted_area = np.log1p(gr_liv_area) * overall_qual

            remaining_cols = X[:, 2:]
            return np.c_[weighted_area, remaining_cols]
        except Exception as ex:
            raise CustomException(ex, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            print(f"R2 score of {list(models.keys())[i]}: {test_model_score}")

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as ex:
        raise CustomException(ex, sys)
