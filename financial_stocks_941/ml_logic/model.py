import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple

# Initialize and compile model
def init_model(X_train, y_train):

    # 1 - RNN architecture
    # ======================
    model = Sequential()
    ## 1.1 - Recurrent Layer
    model.add(LSTM(units = 256,
                          activation='tanh',
                          return_sequences = True,
                          input_shape = (X_train.shape[1],X_train.shape[2]),
                          kernel_regularizer=regularizers.L1L2(l1=0.0001, l2=0.001),
                          recurrent_dropout = 0.3))
    model.add(LSTM(units = 128,
                          activation='tanh',
                          return_sequences = False,
                          recurrent_dropout = 0.35))

    # 1.2 - Predictive Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.45))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=0.45))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(rate=0.50))
    model.add(Dense(1, activation='sigmoid'))

    # 2 - Compiler
    # ======================
    rmsprop =RMSprop(learning_rate=0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics=['accuracy'])

    return model

# Fit model
def fit_model(model: tensorflow.keras.Model, X_train, y_train, verbose=0) -> Tuple[tensorflow.keras.Model, dict]:

    es = EarlyStopping(monitor = "val_loss",
                      patience =100,
                      mode = "auto",
                      restore_best_weights = True)


    history = model.fit(X_train, y_train,
                        validation_split = 0.2,
                        shuffle = False,
                        batch_size = 32,
                        epochs = 600,
                        callbacks = [es],
                        verbose = verbose)

    return model, history
