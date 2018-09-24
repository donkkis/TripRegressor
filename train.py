from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import Sequence
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


class SequenceBatchGenerator(Sequence):
    """
        A generator class to produce a single batch of sequences
        for LSTM training

        Arguments:
            x_set: The whole training set, python list of length m_examples.
            A single example can be accessed in the manner x_set[example_idx]
            and is a numpy array of shape (1, timesteps, n_features). Timesteps
            can vary between examples.

            y_set: The labels corresponding to elements in x_set

            batch_size: The batch size to be used in training

        Outputs:
            batch_x_tensor: Numpy array of shape (batch_size, max_timesteps_batch,
            n_input_features)
            batch_y_tensor: Numpy array of shape (batch_size, max_timesteps_batch,
            n_output_features)


        #https://keras.io/utils/#sequence


    """

    def __init__(self, x_set, y_set, batch_size):
        """
        TODO: Should implement a check that n_features is the same for all examples
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size].copy()

        # get all the stuff required for reshaping
        max_timesteps_batch = max([seq.shape[1] for seq in batch_x])
        input_dim = batch_x[0].shape[2]
        output_dim = batch_y[0].shape[2]

        # initialize return variables as 3D tensors
        batch_x_tensor = np.zeros((len(batch_x), max_timesteps_batch, input_dim))
        batch_y_tensor = np.zeros((len(batch_y), max_timesteps_batch, output_dim))

        # Zero pad all samples within batch to max length
        for i in range(len(batch_x)):
            padding_dims = ((0, 0), (0, max_timesteps_batch - batch_x[i].shape[1]), (0, 0))
            batch_x[i] = np.pad(batch_x[i], padding_dims, 'constant', constant_values=(None, 0))
            batch_y[i] = np.pad(batch_y[i], padding_dims, 'constant', constant_values=(None, 0))

            # Reshape to meet Keras expectation
            batch_x[i][0] = np.reshape(batch_x[i].transpose(), (1, max_timesteps_batch, input_dim))
            batch_y[i][0] = np.reshape(batch_y[i].transpose(), (1, max_timesteps_batch, output_dim))

            # Append x, y to returnable tensor
            batch_x_tensor[i, :, :] = batch_x[i]
            batch_y_tensor[i, :, :] = batch_y[i]

        return batch_x_tensor, batch_y_tensor

def get_model(lstm_layers=4, LSTM_units=200, dropout_rate=0.2):
    # Initialize the RNN
    model = Sequential()

    #LSTM Layers and dropout regularization
    model.add(LSTM(units = LSTM_units, return_sequences=True,
                       input_shape = (None, input_dim)))
    model.add(Dropout(rate = dropout_rate))

    for i in range(layers-1):
        model.add(LSTM(units = LSTM_units, return_sequences=True))
        model.add(Dropout(rate = dropout_rate))

    #Linear output layer
    model.add(TimeDistributed(Dense(1)))

    return model


def train(model, batch_gen, batch_test_gen):
    """

    Args:
        model (keras.models.Sequential:
        batch_gen (SequenceBatchGenerator):
        batch_test_gen (SequenceBatchGenerator:

    Returns:

    """
    # Set up some useful stuff
    from datetime import datetime as dt

    ny = dt.now().strftime("%d-%m-%Y__%H_%M_%S")
    model_outfile = f'{ny}.h5'

    # save_best_only = True --> model_outfile will be overwritten each time val_loss improves
    model_checkpoint = ModelCheckpoint(model_outfile, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(f'{ny}.log')

    # Compile and train the model

    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit_generator(batch_gen, epochs=100, verbose=1, validation_data=batch_test_gen,
                            callbacks=[model_checkpoint, csv_logger])


if __name__ == '__main__':
    # TODO unpack X_train, X_test, y_train, y_test from pickled obj
    X_train, X_test, y_train, y_test = None
    batch_gen = SequenceBatchGenerator(X_train, y_train)
    batch_test_gen = SequenceBatchGenerator(X_test, y_test)
    model = get_model()
    train(model, batch_gen, batch_test_gen)