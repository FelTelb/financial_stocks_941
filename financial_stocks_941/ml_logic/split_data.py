""" Split the data set"""

import numpy as np
import pandas as pd
from typing import Tuple, List
#from ml_logic.params import DATA


# 1) Create FOLDS for cross-validation
def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    '''
    This function slides through the Time Series dataframe of shape (n_days, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold

    Returns a list of folds, each as a 2D - DataFrame
    '''
    # Initialized the folds list
    folds= []
    # Number of strides needed to "see" all the data.
    num_strides = round(len(df)/fold_stride - fold_length/fold_stride) + 1
    # For loop to split the data
    for stride in range(num_strides):
        # Splitting the data based on previously defined # of strides
        fold = df.iloc[stride*fold_stride : stride*fold_stride + fold_length, :]
        # a fold from stride to fold_lenght + fold stride, all the columns!
        folds.append(fold.dropna(axis = 1))
    return folds

# 2) Train/Test Split
def train_test_split(fold:pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
    '''
    Returns a train dataframe and a test dataframe (fold_train, fold_test)
    from which one can sample (X,y) sequences.
    df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    '''
    # Train Fold
    y_train_last = round(len(fold)*train_test_ratio) #Last value of train data.
    fold_train = fold.iloc[0:y_train_last,:]

    # Test Fold
    y_test_first =  y_train_last - input_length # First value of test data.
    fold_test = fold.iloc[y_test_first :, :]

    return (fold_train, fold_test)

# 3) Create (X,y) samples
def get_X_y(fold: pd.DataFrame,
            input_length: int,
            output_length: int,
            sample_stride: int):
    '''
    - slides through a `fold` Time Series (2D array) to create sequences of equal
        * `input_length` for X,
        * `output_length` for y,
    using a temporal gap `sample_stride` between each sequence
    - returns a list of sequences, each as a 2D-array time series
    '''

    X, y = [], []

    for i in range(0, len(fold), sample_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (i + input_length + output_length) > len(fold):
            break
        X_i = fold.drop('trade_flag', axis = 1).iloc[i:i + input_length,:] #drop target column ("trade_flag")
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][['trade_flag']] #  [1:] --> To keep the buckets
        X.append(X_i)
        y.append(y_i)

    X, y = np.array(X), np.array(y)

    return X, y
