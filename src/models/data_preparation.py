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

    train = df[df['Dt'] <= test_split_date]
    test = df[df['Dt'] > test_split_date]

    return train, test


def x_y_split(df: pd.DataFrame, target_cols: list = TARGET_COLS):
    """Split the dataframe into x and y sets.

    Args:
        df (pd.DataFrame): Dataframe to be split.
    """
    y = df[target_cols]
    x = df.drop(target_cols, axis=1)
    return x, y

