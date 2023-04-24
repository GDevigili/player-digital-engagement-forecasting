import random
import pickle

# Data manipulation
import pandas as pd
import numpy as np

# model evaluation
from sklearn.metrics import mean_absolute_error

# constants
PROCESSED_DATA_PATH = '../data/processed-data/'
MODEL_PATH = '../models/trained-models/'
TARGET_COLS = ['target1', 'target2', 'target3', 'target4']
TEST_SPLIT_DATE = '2021-04-30'



def naive(test):
    """Naive model that predicts the previous value of the target column"""
    y_pred = pd.DataFrame(columns=TARGET_COLS)
    for target in TARGET_COLS:
        y_pred[target] = test[target + '_shift_1']
    return y_pred


class MeanModel():
    """Mean model that predicts the mean of the target column 
    for each player"""
    def __init__(self, target_cols = TARGET_COLS):
        pass

    def fit(self, X):
        self.player_mean = X.groupby('IdPlayer').mean()

    def predict(self, X):
        y_pred = pd.DataFrame(columns=TARGET_COLS)
        for target in TARGET_COLS:
            y_pred[target] = X['IdPlayer'].map(self.player_mean[target])
            y_pred[target].fillna(self.player_mean[target].mean())
        return y_pred


def evaluate_mae(y_true, y_pred):
    """Evaluate the mean absolute error for each target column

    Parameters
    ----------
    y_true : pd.DataFrame
        True labels
    y_pred : pd.DataFrame
        Predictions
    
    Returns
    -------
    dict
        Mean absolute error for each target column
    """
    maes = {}
    for target in TARGET_COLS:
        mae = mean_absolute_error(y_true[target], y_pred[target])
        maes[target] = mae
    return maes


def fit_predict_targets(model, x_train, y_train, x_test, target_cols=TARGET_COLS, return_models=False):
    """Fit the model and predict for each target column
    
    Parameters
    ----------
    model : sklearn model
        Model to fit and predict
    x_train : pd.DataFrame
        Training data
    y_train : pd.DataFrame
        Training labels
    x_test : pd.DataFrame
        Test data
    target_cols : list, optional
        List of target columns, by default TARGET_COLS
        
    Returns
    -------
    pd.DataFrame
        Predictions for each target column
    """
    y_preds = pd.DataFrame(columns=target_cols)
    models = []
    for target in target_cols:
        model.fit(x_train, y_train[target])
        y_preds[target] = model.predict(x_test)
        if return_models:
            models.append(model)
        else:
            del model
    if return_models:
        return y_preds, models
    else:
        return y_preds
    
    