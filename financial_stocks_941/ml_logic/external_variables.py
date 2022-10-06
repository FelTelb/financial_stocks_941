# Technical indicators

from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_technical_indicators(target_alpha : str,
                             alpha_train: pd.DataFrame,
                             alpha_test : pd.DataFrame)-> Tuple[pd.DataFrame]:

    '''
    Returns 2 DataFrames alpha_train and alpha_test with the technical indicators calculated
    Note: 20 elements are removed from alpha TRAIN.
    '''

    alpha_train_index = alpha_train.index
    alpha_test_index = alpha_test.index

    # Concatenate alpha train and alpha test
    dataset = pd.concat([alpha_train, alpha_test], ignore_index = True, axis = 0)
    len_alpha_train = len(alpha_train)

    # Create 7 and 21 days Simple Moving Average
    dataset['sma7'] = dataset[target_alpha].rolling(window=7).mean()
    dataset['sma21'] = dataset[target_alpha].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset[target_alpha].ewm(span=26).mean() #26-day ema of closing price
    dataset['12ema'] = dataset[target_alpha].ewm(span=12).mean() #12-day ema of closing price
    dataset['MACD'] = (dataset['12ema']-dataset['26ema']) #MACD
    dataset['MACD_S'] = dataset['MACD'].ewm(span=9).mean() #9-day ema of MACD --> trigger line (signal)
    dataset['MACD_h'] = dataset['MACD'] - dataset['MACD_S'] # Convergence/divergence value

    # Create Bollinger Bands
    dataset['20std'] = dataset[target_alpha].rolling(20).std() #STD of last 20 days
    dataset['upper_band'] = dataset['sma21'] + (dataset['20std']*2) # Upper band
    dataset['lower_band'] = dataset['sma21'] - (dataset['20std']*2) # Lower band

    # Create Momentum
    dataset['momentum'] = dataset[target_alpha]-1

    # Clean technical inndicators dataframe
    dataset = dataset.drop(columns = ['26ema', '12ema','20std'])

    alpha_train = dataset.iloc[0:len_alpha_train , :]
    alpha_test = dataset.iloc[len_alpha_train :, :]

    alpha_train = alpha_train.set_index(keys = alpha_train_index).dropna()
    alpha_test = alpha_test.set_index(keys = alpha_test_index)

    return (alpha_train, alpha_test)

#Get external and fundamentals
def get_df_alpha (ext_fund_var : pd.DataFrame,
                  nb_fundamentals: int,
                  alpha_train : pd.DataFrame,
                  alpha_test : pd.DataFrame)-> Tuple[pd.DataFrame]:

    '''
    Returns 2 DataFrames alpha_train and alpha_test with the fundamental indicators and the comodities
    from which one can split the data into X and y samples.
    '''

    # Get a list with the date indexes
    train_index = alpha_train.index
    test_index = alpha_test.index

    # Split the external_fundamentals_variables into train and test
    ext_fund_train = ext_fund_var.loc[train_index,:].iloc[:,nb_fundamentals:] # First loc to get train and test data and second .iloc to remove the fund columns
    ext_fund_test = ext_fund_var.loc[test_index,:].iloc[:,nb_fundamentals:] # First loc to get train and test data and second .iloc to remove the fund columns

    # Get the fundamentals data.
    fund_train = ext_fund_var.loc[train_index,:].iloc[:,:nb_fundamentals]
    fund_test = ext_fund_var.loc[test_index,:].iloc[:,:nb_fundamentals]

    columns = ext_fund_train.columns

    # Initialize the scaler.
    scaler_ext = StandardScaler().fit(ext_fund_train)
    # Scale the data
    ext_train_scaled = scaler_ext.transform(ext_fund_train)
    ext_train_scaled = pd.DataFrame(ext_train_scaled, columns = columns).set_index(keys = train_index)

    ext_test_scaled = scaler_ext.transform(ext_fund_test)
    ext_test_scaled = pd.DataFrame(ext_test_scaled, columns = columns).set_index(keys = test_index)

    # Merge the datasets alpha train + externals scaled and fundamentals.
    alpha_train_ext = pd.merge(alpha_train, ext_train_scaled, how ='left' , on = 'date')
    alpha_test_ext = pd.merge(alpha_test, ext_test_scaled, how ='left' , on = 'date')

    alpha_train_ext_fund = pd.merge(alpha_train_ext, fund_train, how ='left' , on = 'date')
    alpha_test_ext_fund = pd.merge(alpha_test_ext, fund_test, how ='left' , on = 'date')

    return (alpha_train_ext_fund, alpha_test_ext_fund)
