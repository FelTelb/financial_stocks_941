import os

## DIR PARAMS
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SP500_R_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'sp500_return.csv')
EXT_VAR_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'ext_fund_var.csv')

# --------------------------------------------------- #
# Target's column name.                               #
# --------------------------------------------------- #
TARGET_RETURN = 'R_BAC'
TARGET_ALPHA = 'A_BAC'
# --------------------------------------------------- #
# Number of fundamental variables considered.         #
# --------------------------------------------------- #
NB_FUNDAMENTALS = 2
# --------------------------------------------------- #
# Number of PCA components to keep.                   #
# --------------------------------------------------- #
PCA_COMPONENTS = 3
# --------------------------------------------------- #
# FOLDS with a length of X days.                      #
# --------------------------------------------------- #
FOLD_LENGTH = 1000
# --------------------------------------------------- #
# Fold's stride between each fold.                    #
# --------------------------------------------------- #
FOLD_STRIDE = 125
# --------------------------------------------------- #
# Train-test-split ratio.                             #
# --------------------------------------------------- #
TRAIN_TEST_RATIO = 0.7
# --------------------------------------------------- #
# Number of Days used for prediction.                 #
# --------------------------------------------------- #
INPUT_LENGTH = 20
# --------------------------------------------------- #
# Number of days to be precited.                      #
# --------------------------------------------------- #
OUTPUT_LENGTH = 1
# --------------------------------------------------- #
# Sample's stride.                                    #
# --------------------------------------------------- #
SAMPLE_STRIDE = 1
