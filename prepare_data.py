import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEQID_COL = 'sequence'


def normalize(data: pd.DataFrame, outfile=None) -> pd.DataFrame:
    """
    Args:
        data (pandas.DataFrame) : DF to be normalized
    Returns:
        data (pandas.DataFrame) : Normalized df except for SEQID_COL column
    """
    sc = StandardScaler()

    # Ignore categorical columns
    to_be_normalized = list(data.select_dtypes(include=np.number).keys())
    to_be_normalized.remove(SEQID_COL)

    data[to_be_normalized] = sc.fit_transform(data[to_be_normalized])

    if outfile:
        with open(outfile, 'wb') as f:
            pickle.dump(sc, f)
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


def prepare_dataset(data, target_cols, outfile_scaler=None, outfile='./data/dump.pickle', outfile_val='./data/dump_val.pickle', exclude=[], order=True, normalize_data=True, test_size=0.05):
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
        data = normalize(data, outfile_scaler)
    data = data_to_list(data)
    if order:
        data = order_sequences_by_length(data)

    try:
        X = [seq.drop(target_cols + exclude, axis=1) for seq in data]
        y = [pd.DataFrame(seq[target_cols]) for seq in data]
    except ValueError:
        raise ValueError("Column not found")

    X, y = dataset_to_numpy(X, y)

    # Train / test / validate split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    data = (X_train, X_test, y_train, y_test)

    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

    with open(outfile_val, 'wb') as f:
        pickle.dump((X_validate, y_validate), f)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, required=True,
                        help='Path to an object that can be parsed as pandas DataFrame')
    parser.add_argument('-e', '--exclude', type=str, nargs='*', default=[],
                        help='names of the columns to be dropped from the final dataset')
    parser.add_argument('-test', '--test_size', type=float, default=0.3,
                        help='float between 0 ... 1, the test set proportion')
    parser.add_argument('-or', '--order_dataset', type=bool, default=False,
                        help='order the resulting dataset to ascending order')
    parser.add_argument('-n', '--normalize_data', type=bool, default=True,
                        help='normalize data using StandardScaler')
    parser.add_argument('-t', '--target_columns', type=str, nargs='+', required=True,
                        help='column to be predicted')
    parser.add_argument('-s', '--seqid_col', type=str, required=True,
                        help='Column where sequence id is located')
    parser.add_argument('-o', '--out_file', type=str, required=True,
                        help='path for pickled object output')
    parser.add_argument('-ov', '--out_file_validate', type=str, required=True,
                        help='path for pickled validation object output')
    parser.add_argument('-sco', '--out_file_scaler', type=str, required=True,
                        help='output path for MinMaxScaler object')

    FLAGS, unparsed = parser.parse_known_args()
    SEQID_COL = FLAGS.seqid_col

    data = pd.read_csv(FLAGS.file_path)
    target_cols = FLAGS.target_columns
    exclude = FLAGS.exclude
    order = FLAGS.order_dataset
    normalize_data = FLAGS.normalize_data
    test_size = FLAGS.test_size
    outfile = FLAGS.out_file
    outfile_val = FLAGS.out_file_validate
    outfile_scaler = FLAGS.out_file_scaler

    prepare_dataset(data,
                    target_cols,
                    outfile_scaler,
                    outfile,
                    outfile_val,
                    exclude,
                    order,
                    normalize_data,
                    test_size)