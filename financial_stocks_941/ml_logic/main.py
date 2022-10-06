import numpy as np
import pandas as pd
import os
from financial_stocks_941.ml_logic.split_data import get_folds, get_X_y, train_test_split
from financial_stocks_941.ml_logic.pseudo_alpha import get_spx_df, get_data_scaled, get_projections, get_alpha
from financial_stocks_941.ml_logic.params import FOLD_LENGTH, FOLD_STRIDE, PCA_COMPONENTS, SP500_R_CSV_PATH
from financial_stocks_941.ml_logic.params import TRAIN_TEST_RATIO, INPUT_LENGTH, TARGET_ALPHA
import datetime

def parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')

def test(sp_500_df):
    folds = get_folds (sp_500_df, FOLD_LENGTH, FOLD_STRIDE)
    fold = folds[7]
    # One fold train split, one fold test
    (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
    # Train_SP500, Test_SP500, Train_SPX, Test_SPX
    (fold_train_500, fold_test_500) = get_spx_df(fold_train, fold_test)
    # Scale data
    (ftrain_500_scaled, ftest_500_scaled) = get_data_scaled(fold_train_500,fold_test_500)
    # Project data via PCA
    (projections_train, projections_test) = get_projections(PCA_COMPONENTS,
                                              ftrain_500_scaled,
                                              ftest_500_scaled)
    # Compute Alpha
    (alpha_train, alpha_test) = get_alpha (
                                      TARGET_ALPHA,
                                      ftrain_500_scaled,
                                      ftest_500_scaled,
                                      projections_train,
                                      projections_test
                                      )
    return alpha_train

if __name__ == '__main__':
    try:
        sp_500_df = pd.read_csv(SP500_R_CSV_PATH,
                                header=0, parse_dates=[0], date_parser=parser).iloc[1:,:].set_index("date")
        print ((test(sp_500_df)).shape)
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
