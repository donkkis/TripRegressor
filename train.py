from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import Sequence
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from datetime import datetime as dt
import numpy as np
import argparse
import pickle
import os
import tensorflow as tf

FLAGS = None


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

    def __init__(self, x_set, y_set, batch_size=128):
        """
        """
        self.batch_size = batch_size

        # Make sure n_features is same for all examples
        unique_input_dims = len(set([example.shape[2] for example in x_set]))
        unique_output_dims = len(set([example.shape[2] for example in y_set]))
        if not unique_input_dims == unique_output_dims == 1:
            raise Exception("n_features needs to be same for all examples")

        self.input_dim = x_set[0].shape[2]
        self.output_dim = y_set[0].shape[2]

        # TODO Refactor into a function and add unit test
        # make sure sequences are ascending by length to reduce padding
        # get the lenghts of the elements in x_train
        lengths = np.array([[idx, trip.shape[1]] for idx, trip in enumerate(x_set)])
        lengths = lengths[lengths[:, 1].argsort()]
        # order x_train, y_train to ascending order by sequence length
        idx = lengths[:, 0].tolist()

        x_set = [x_set[i] for i in idx]
        y_set = [y_set[i] for i in idx]

        self.x, self.y = x_set, y_set

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size].copy()

        # get all the stuff required for reshaping
        max_timesteps_batch = max([seq.shape[1] for seq in batch_x])

        # initialize return variables as 3D tensors
        batch_x_tensor = np.zeros((len(batch_x), max_timesteps_batch, self.input_dim))
        batch_y_tensor = np.zeros((len(batch_y), max_timesteps_batch, self.output_dim))

        # Zero pad all samples within batch to max length
        for i in range(len(batch_x)):
            padding_dims = ((0, 0), (0, max_timesteps_batch - batch_x[i].shape[1]), (0, 0))
            batch_x[i] = np.pad(batch_x[i], padding_dims, 'constant', constant_values=(None, 0))
            batch_y[i] = np.pad(batch_y[i], padding_dims, 'constant', constant_values=(None, 0))

            # Reshape to meet Keras expectation
            batch_x[i][0] = np.reshape(batch_x[i].transpose(), (1, max_timesteps_batch, self.input_dim))
            batch_y[i][0] = np.reshape(batch_y[i].transpose(), (1, max_timesteps_batch, self.output_dim))

            # Append x, y to returnable tensor
            batch_x_tensor[i, :, :] = batch_x[i]
            batch_y_tensor[i, :, :] = batch_y[i]

        return batch_x_tensor, batch_y_tensor


def get_model(input_dim, lstm_layers=1, lstm_units=50, dropout_rate=0.2):
    # Initialize the RNN
    model = Sequential()

    #LSTM Layers and dropout regularization
    model.add(LSTM(units = lstm_units, return_sequences=True,
                   input_shape = (None, input_dim)))
    model.add(Dropout(rate = dropout_rate))

    for i in range(lstm_layers-1):
        model.add(LSTM(units = lstm_units, return_sequences=True))
        model.add(Dropout(rate = dropout_rate))

    #Linear output layer
    model.add(TimeDistributed(Dense(1)))

    return model


def mae_exclude_padding(y_true, y_pred):
    """

    Args:
        y_true, y_pred: tf.Tensors of shape (batch_size, max_timesteps, output_dim)
    Returns:
    """
    y_mask = tf.not_equal(y_true, tf.constant(0, dtype=tf.float32))
    y_true_masked = tf.boolean_mask(y_true, y_mask)
    y_pred_masked = tf.boolean_mask(y_pred, y_mask)

    error = tf.reduce_mean(tf.abs(tf.subtract(y_true_masked, y_pred_masked)))

    return error

def mse_exclude_padding(y_true, y_pred):
    """

    Args:
        y_true, y_pred: tf.Tensors of shape (batch_size, max_timesteps, output_dim)
    Returns:
    """
    y_mask = tf.not_equal(y_true, tf.constant(0, dtype=tf.float32))
    y_true_masked = tf.boolean_mask(y_true, y_mask)
    y_pred_masked = tf.boolean_mask(y_pred, y_mask)

    error = tf.reduce_mean(tf.square(tf.subtract(y_true_masked, y_pred_masked)))

    return error

def train():
    """

    Args:

    Returns:

    """

    def get_first_file(path):
        filename = os.listdir(path)[0]
        return os.path.join(path, filename)

    # Set up some useful stuff

    if not FLAGS.file_path:
        INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
        INPUTS_DIR = os.path.join(INPUTS_DIR, 'training-data')
        FLAGS.file_path = get_first_file(INPUTS_DIR)

    X_train, X_test, y_train, y_test = pickle.load(open(FLAGS.file_path, 'rb'))
    batch_gen = SequenceBatchGenerator(X_train, y_train)
    batch_test_gen = SequenceBatchGenerator(X_test, y_test)

    # TODO Should check equal input_dim for X_test and X_train?
    model = get_model(input_dim=batch_gen.input_dim, lstm_layers=FLAGS.lstm_layers,
                      lstm_units=FLAGS.units_per_layer, dropout_rate=FLAGS.dropout_rate)
    val_steps = np.floor(len(X_test) / FLAGS.batch_size_val)
    train_steps = np.floor(len(X_train) / FLAGS.batch_size_train)

    tic = dt.now().strftime("%d-%m-%Y__%H_%M_%S")
    model_outfile = '{}.h5'.format(tic)

    # Initialize callbacks
    # save_best_only = True --> model_outfile will be overwritten each time val_loss improves
    model_checkpoint = ModelCheckpoint(model_outfile, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('{}.log'.format(tic))
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0,
                                   mode='min')

    # Compile and train the model
    model.compile(optimizer='adam', loss=mae_exclude_padding)
    model.fit_generator(batch_gen,
                        epochs=FLAGS.epochs,
                        verbose=1,
                        validation_data=batch_test_gen,
                        use_multiprocessing=False,
                        callbacks=[model_checkpoint, csv_logger, early_stopping])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=False,
                        help='Path to a pickled object containing X_train, X_test, y_test, y_train')
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help='Max number of epochs to train')
    parser.add_argument('-p', '--patience', type=int, required=True,
                        help='Epoch patience parameter for early stopping')
    parser.add_argument('-l', '--lstm_layers', type=int, required=True,
                        help='Number of LSTM layers')
    parser.add_argument('-u', '--units_per_layer', type=int, required=True,
                        help='Number of LSTM units per layer')
    parser.add_argument('-k', '--dropout_rate', type=float, default=0.0,
                        help='Reciprocal dropout probability (default = 1.0)')
    parser.add_argument('-bt', '--batch_size_train', type=int, required=True,
                        help='Training batch size')
    parser.add_argument('-bv', '--batch_size_val', type=int, required=True,
                        help='Validation batch size')
    FLAGS, unparsed = parser.parse_known_args()
    train()
