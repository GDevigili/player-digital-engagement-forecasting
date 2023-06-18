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


# Funções auxiliares para carregar e tratar os dados
def unpack_json(json_str):
    return pd.DataFrame() if pd.isna(json_str) else pd.read_json(json_str)

def unpack_data(data, dfs=None, n_jobs=-1):
    if dfs is not None:
        data = data.loc[:, dfs]
    unnested_dfs = {}
    for name, column in data.iteritems():
        daily_dfs = Parallel(n_jobs=n_jobs)(
            delayed(unpack_json)(item) for date, item in column.iteritems())
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