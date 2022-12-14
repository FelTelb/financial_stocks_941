import pandas as pd
from financial_stocks_941.ml_logic.split_data import get_folds, get_X_y, train_test_split
from financial_stocks_941.ml_logic.pseudo_alpha import get_spx_df, get_data_scaled, get_projections, get_alpha, get_bucket
from financial_stocks_941.ml_logic.params import FOLD_LENGTH, FOLD_STRIDE, PCA_COMPONENTS, SP500_R_CSV_PATH
from financial_stocks_941.ml_logic.params import TRAIN_TEST_RATIO, INPUT_LENGTH, TARGET_ALPHA, EXT_VAR_CSV_PATH
from financial_stocks_941.ml_logic.params import NB_FUNDAMENTALS, OUTPUT_LENGTH, SAMPLE_STRIDE
from financial_stocks_941.ml_logic.external_variables import get_df_alpha
from financial_stocks_941.ml_logic.model import init_model, fit_model
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
    # Get buckets
    (alpha_train, alpha_test) = get_bucket (
                                        TARGET_ALPHA,
                                        alpha_train,
                                        alpha_test
                                      )
    # Get externals and fundamentals
    (alpha_train_ext_fund, alpha_test_ext_fund) = get_df_alpha (
                                          ext_fund_var,
                                          NB_FUNDAMENTALS,
                                          alpha_train,
                                          alpha_test
                                          )
    # Get train and test samples
    X_train, y_train = get_X_y(alpha_train_ext_fund, INPUT_LENGTH, OUTPUT_LENGTH, SAMPLE_STRIDE)
    X_test, y_test = get_X_y(alpha_test_ext_fund, INPUT_LENGTH, OUTPUT_LENGTH, SAMPLE_STRIDE)

    # 2 - Modelling
    # =========================================
    ##### LSTM Model
    model = init_model(X_train, y_train)
    model, history = fit_model(model, X_train, y_train)
    res = model.evaluate(X_test, y_test)[1]

    return res

if __name__ == '__main__':
    try:
        sp_500_df = pd.read_csv(SP500_R_CSV_PATH,
                                header=0, parse_dates=[0], date_parser=parser).iloc[1:,:].set_index("date")
        ext_fund_var = pd.read_csv(EXT_VAR_CSV_PATH,
                                   parse_dates = ["date"]).set_index("date")
        print (test(sp_500_df))
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
