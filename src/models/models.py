import random
import pickle

# Data manipulation
import pandas as pd
import numpy as np

# constants
PROCESSED_DATA_PATH = '../data/processed-data/'
MODEL_PATH = '../models/trained-models/'
TARGET_COLS = ['target1', 'target2', 'target3', 'target4']
TEST_SPLIT_DATE = '2021-04-30'


def naive(test):
    y_pred = pd.DataFrame(columns=TARGET_COLS)
    for target in TARGET_COLS:
        y_pred[target] = test[target + '_shift_1']
    return y_pred


class MeanModel():
    def __init__(self, target_cols = TARGET_COLS):
        pass

    def fit(self, X):
        self.player_mean = X.groupby('IdPlayer').mean()

    def predict(self, X):
        y_pred = pd.DataFrame(columns=TARGET_COLS)
        for target in TARGET_COLS:
            y_pred[target] = X['IdPlayer'].map(self.player_mean[target])
        return y_pred
