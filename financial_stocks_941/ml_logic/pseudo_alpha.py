from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Separate S&P500 and SPX returns
def get_spx_df(
    fold_train : pd.DataFrame,
    fold_test : pd.DataFrame) -> Tuple[pd.DataFrame]:
    '''
    Returns 4 dataframes:
    (fold_train_500 -> returns train_df 500 stocks,
    fold_test_500 -> returns test_df 500 stocks
    from which one can scale the data.
    '''
    # Fold Train
    fold_train_500 = fold_train.dropna(axis=1).drop(columns = "R_^GSPC")
    stocks = fold_train_500.columns

    # Fold Test
    fold_test_500 = fold_test.loc[: ,stocks]

    return (fold_train_500, fold_test_500)

# Scale the data
def get_data_scaled(fold_train_500 : pd.DataFrame,
                    fold_test_500 : pd.DataFrame) -> Tuple[pd.DataFrame]:

    '''
    Returns 2 scaled dataframes:
    (ftrain_500_scaled -> returns a scaled train_df 500 stocks,
    ftest_500_scaled -> returns a scaled test_df 500 stocks
    from which one can apply a PCA
    '''
    stocks = fold_train_500.columns
    index_train = fold_train_500.index
    index_test = fold_test_500.index


    # Scaling 500 stocks
    scaler_500 = StandardScaler().fit(fold_train_500)

    ftrain_500_scaled = scaler_500.transform(fold_train_500)
    ftrain_500_scaled = pd.DataFrame(ftrain_500_scaled, columns = stocks).set_index(keys = index_train)

    ftest_500_scaled = scaler_500.transform(fold_test_500)
    ftest_500_scaled = pd.DataFrame(ftest_500_scaled, columns = stocks).set_index(keys = index_test)

    return (ftrain_500_scaled, ftest_500_scaled)

# Calculate projections
def get_projections(pca_components: int,
                    ftrain_500_scaled : pd.DataFrame,
                    ftest_500_scaled : pd.DataFrame,
                    ) -> Tuple[pd.DataFrame] :
    '''
      Returns 2 scaled projected dataframes:
      (projections_train -> train PCA projections,
      projections_test -> test PCA projections,
      from which one can calculate an alpha
      '''

    ftrain_500_scaled_index = ftrain_500_scaled.index
    ftest_500_scaled_index = ftest_500_scaled.index

    # Instantiating the PCA
    pca_train = PCA(n_components = pca_components)

    pca_train.fit(ftrain_500_scaled)
    projections_train = pd.DataFrame(pca_train.transform(ftrain_500_scaled))

    projections_test = pd.DataFrame(pca_train.transform(ftest_500_scaled))

    # Scaling the projections
    scaler_proj = StandardScaler().fit(projections_train)

    projections_train = scaler_proj.transform(projections_train)

    projections_test = scaler_proj.transform(projections_test)

    # Renaming DataFrames
    projections_train= pd.DataFrame(projections_train,
                    columns=[f'PC_proj_{i+1}' for i in range(pca_components)])\
                    .set_index(keys = ftrain_500_scaled_index)

    projections_test= pd.DataFrame(projections_test,
                columns=[f'PC_proj_{i+1}' for i in range(pca_components)])\
                    .set_index(keys = ftest_500_scaled_index)

    return (projections_train, projections_test)

# Calculate pseudo-alpha
def get_alpha (target_alpha: str,
               ftrain_500_scaled: pd.DataFrame,
               ftest_500_scaled: pd.DataFrame,
               projections_train: pd.DataFrame,
               projections_test: pd.DataFrame
               ) -> Tuple[pd.DataFrame]:

    '''
      Returns 2 dataframes:
      (alpha_train -> train PCA projections,
      alpha_test -> test PCA projections,
      from which one can calculate an alpha
      '''

    # Alpha calulation
    alpha_train = ftrain_500_scaled.add(projections_train.iloc[:,0], axis = 0) # Substracting first column of projections
    alpha_test = ftest_500_scaled.add(projections_test.iloc[:,0], axis = 0) # Substracting first column of projections

    # Rename the columns
    alpha_names = [f'A{stock[1:]}' for stock in alpha_train.columns] # R_<Stock name> to A_<Stock name>
    alpha_train = alpha_train.set_axis(alpha_names, axis = 1).copy() # Rename alpha_train
    alpha_test = alpha_test.set_axis(alpha_names, axis = 1).copy() # Rename alpha_test

    return (alpha_train[[target_alpha]], alpha_test[[target_alpha]])
