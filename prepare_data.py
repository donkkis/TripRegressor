import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEQID_COL = 'sequence'


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        data (pandas.DataFrame) : DF to be normalized
    Returns:
        data (pandas.DataFrame) : Normalized df except for SEQID_COL column
    """
    sc = StandardScaler()

    to_be_normalized = list(data.keys())
    to_be_normalized.remove(SEQID_COL)

    data[to_be_normalized] = sc.fit_transform(data[to_be_normalized])

    return data


def data_to_list(data):
    """

    Args:
        data (pandas.DataFrame) : DataFrame containing a sequence column
        seqid_col (string) : The column where sequenceId is located

    Returns:
        data (list) : The equivalent data so that each sequence has been
            cast as an pandas.DataFrame entry in a python list
    """
    seq_ids = list(data[SEQID_COL].unique())
    data = [data[data[SEQID_COL] == i] for i in seq_ids]
    return data


def order_sequences_by_length(data):
    """
    Args:
        data (list) : list of pandas.DataFrame instances to be ordered
    Returns:
        data (list) : the transformed data in ascending order by sequence lenght
    """
    lenghts = list(map(len, data))
    data = list(zip(lenghts, data))
    data.sort(key=lambda tup: tup[0])
    data = [tup[1] for tup in data]

    return data


def dataset_to_numpy(X, y):
    """
    Args:
        X (list) : a python list of pd.DataFrame instances
        y (list) : a python list of pd.DataFrame instances


    Returns:
        X (list), y (list) : dataset converted to list numpy arrays of length m_examples.
            A single example should be accessable in the manner x_set[example_idx]
            and is a numpy array of shape (1, timesteps, n_features), where n_features is the
            input/output dimension of the model, respectively. Furthermore, number of timesteps
            is allowed to vary between examples.
    """

    X = list(map(lambda ex: np.array(ex, ndmin=3), X))
    y = list(map(lambda ex: np.array(ex, ndmin=3), y))
    return X, y


def prepare_dataset(data, target_cols, exclude=[], order=True, normalize_data=True, test_size=0.05):
    """

    Args:
        data (pandas.DataFrame) : the sequential data to be prepared
        target_cols (list str) : the name(s) of the dependent variable(s)
        exclude (list str) : the name(s) of columns to be dropped from the
            resulting final dataset
        test_size (float) : float between 0...1, the test set proportion

    Returns:
        X, y : python lists of numpy arrays
    """

    if normalize_data:
        data = normalize(data)
    data = data_to_list(data)
    if order:
        data = order_sequences_by_length(data)

    try:
        X = [seq.drop(target_cols + exclude, axis=1) for seq in data]
        y = [pd.DataFrame(seq[target_cols]) for seq in data]
    except ValueError:
        raise ValueError("Column not found")

    X, y = dataset_to_numpy(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    data = (X_train, X_test, y_train, y_test)

    with open('./data/data.pickle', 'wb') as f:
        pickle.dump(data, f)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=True,
                        help='Path to an object that can be parsed as pandas DataFrame')
    parser.add_argument('-e', '--exclude', type=str, nargs='*',
                        help='names of the columns to be dropped from the final dataset')
    parser.add_argument('-test', '--test_size', type=float, default=0.2,
                        help='float between 0 ... 1, the test set proportion')
    parser.add_argument('-or', '--order_dataset', type=bool, default=False,
                        help='order the resulting dataset to ascending order')
    parser.add_argument('-n', '--normalize_data', type=bool, default=True,
                        help='normalize data using StandardScaler')
    parser.add_argument('-t', '--target_columns', type=str, nargs='+', required=True,
                        help='column to be predicted')
    parser.add_argument('-s', '--seqid_col', type=str, required=True,
                        help='Column where sequence id is located')
    FLAGS, unparsed = parser.parse_known_args()
    SEQID_COL = FLAGS.seqid_col
    print(FLAGS)