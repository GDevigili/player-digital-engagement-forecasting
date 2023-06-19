import random
import pickle

# Data manipulation
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# constants
PROCESSED_DATA_PATH = '../data/processed-data/'
MODEL_PATH = '../models/trained-models/'
TARGET_COLS = ['target1', 'target2', 'target3', 'target4']
TEST_SPLIT_DATE = '2021-04-30'


# Funções auxiliares para carregar e tratar os dados
def unpack_json(json_str):
    return pd.DataFrame() if pd.isna(json_str) else pd.read_json(json_str)

def unpack_data(data, dfs=None, n_jobs=-1):
    if dfs is not None:
        data = data.loc[:, dfs]
    unnested_dfs = {}
    for name, column in data.iteritems():
        daily_dfs = Parallel(n_jobs=n_jobs)(delayed(unpack_json)(item) for date, item in column.iteritems())
        df = pd.concat(daily_dfs)
        unnested_dfs[name] = df
    return unnested_dfs

def create_id(df, id_cols, id_col_name, dt_col_name = 'Dt'):
    df['Id' + dt_col_name + id_col_name] = df[id_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    return df


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and is_datetime(df[col]) == False and col_type != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif is_datetime(df[col]) == True:
            df[col] = df[col].astype('datetime64[ns]')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Funções auxiliares para o pré-processamento dos dados
def sort_df(df: pd.DataFrame, columns: list = ['IdPlayer', 'Dt']) -> None:
    """Sort the dataframe by the columns passed as argument.
    
    Args:
        df (pd.DataFrame): Dataframe to be sorted.
        columns (list, optional): Columns to sort the dataframe. Defaults to ['IdPlayer', 'Dt'].
        
        Returns:
            None
    """
    df.sort_values(by=columns, inplace=True)
    # reset index
    df.reset_index(drop=True, inplace=True)


def shift_targets(df, shift_vals: list = [1, 2, 3, 4, 5, 6, 7, 14, 30]):
    """Shift the targets by the values passed as argument.

    Args:
        df (pd.DataFrame): Dataframe to be shifted.
        shift_vals (list, optional): Values to shift the targets. Defaults to [1, 2, 3, 4, 5, 6, 7, 14, 30].

    Returns:
        pd.DataFrame: Dataframe with the shifted targets.
    """
    df_aux = pd.DataFrame()
    # Iterate over players to make the shift only using the player data
    for player in df['IdPlayer'].unique():
        df_player = df[df['IdPlayer'] == player]
        # Iterate over the pre-defined shift values
        for shift_val in shift_vals:
            # Iterate over the targets
            for target in TARGET_COLS:
                # Make the shift
                df_player[f'{target}_shift_{shift_val}'] = df_player[target].shift(shift_val)
        # Concatenate the player data with the rest of the data
        df_aux = pd.concat([df_aux, df_player], axis=0)
        # Remove the player data from memory
        del df_player
    # df.dropna(inplace=True)
    return df_aux


def train_test_split(
    df: pd.DataFrame
    ,test_split_date: str = TEST_SPLIT_DATE
    ):
    """Split the dataframe into train and test sets.

    Args:
        df (pd.DataFrame): Dataframe to be split.
        test_split_date (str, optional): Date to split the dataframe. Defaults to TEST_SPLIT_DATE.
    """

    train = df[(df.Dt <= "2021-01-31") & (df.Dt >= "2018-01-01")] 
    val = df[(df.Dt <= "2021-04-30") & (df.Dt >= "2021-02-01")] 
    test = df[(df.Dt <= "2021-07-31") & (df.Dt >= "2021-05-01")]
    # train.to_csv('train.csv', index=None)
    # val.to_csv('validation.csv', index=None) 
    # test.to_csv('test.csv', index=None) 

    return train, test, val


def x_y_split(df: pd.DataFrame, target_cols: list = TARGET_COLS):
    """Split the dataframe into x and y sets.

    Args:
        df (pd.DataFrame): Dataframe to be split.
    """
    y = df[target_cols]
    x = df.drop(target_cols, axis=1)
    return x, y

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
    maes['average | MAE'] = np.mean(list(maes.values()))
    return maes


def MAE(y_pred, y_obs):
    return np.mean(np.abs(y_pred - y_obs))

def AMAE(y_obs, y_pred, points=1000, show=True):
    limits = np.linspace(0, 100, points)
    dx = limits[1] - limits[0]
    maes = []
    for x in limits:
        inx = y_obs >= x
        maes.append(MAE(np.array(y_pred)[inx], np.array(y_obs)[inx]))

    if show:
        import matplotlib.pyplot as plt
        plt.plot(limits, maes)
        plt.xlabel('Threshold')
        plt.ylabel('MAE')
        plt.show()

    return np.sum(maes) * dx

def evaluate_amae(y_true, y_pred):
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
    amaes = {}
    for target in TARGET_COLS:
        amae = AMAE(y_true[target], y_pred[target], show=False)
        amaes[target + ' | AMAE'] = amae
    amaes['average | AMAE'] = np.mean(list(amaes.values()))
    return amaes

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
    
    