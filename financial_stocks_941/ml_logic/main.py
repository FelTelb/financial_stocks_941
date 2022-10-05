import numpy as np
import pandas as pd
import os
from financial_stocks_941.ml_logic.split_data import get_folds, get_X_y, train_test_split
from financial_stocks_941.ml_logic.params import FOLD_LENGTH, FOLD_STRIDE, SP500_R_CSV_PATH
from financial_stocks_941.ml_logic.params import TRAIN_TEST_RATIO, INPUT_LENGTH

def train(sp_500_df):
    folds = get_folds (sp_500_df, FOLD_LENGTH, FOLD_STRIDE)
    fold = folds[7]
    # One fold train split, one fold test
    (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
    return fold_train

if __name__ == '__main__':
    sp_500_df = pd.read_csv(SP500_R_CSV_PATH)
    #print ((train(sp_500_df)).shape)
