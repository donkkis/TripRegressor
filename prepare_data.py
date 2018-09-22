import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def data_to_list(data, seqid_col):
    """

    Args:
        data (pandas.DataFrame) : DataFrame containing a sequence column
        seqid_col (string) : The column where sequencId is located

    Returns:
        data (list) : The equivalent data so that each sequence has been
            cast as an entry in a python list
    """
    seq_ids = list(data[seqid_col].unique())
    data = [data[data[seqid_col] == i]  for i in seq_ids]
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


def prepare_dataset(data, target_cols, test_size=0.05):
    """

    Args:
        data (pandas.DataFrame) : the sequential data to be prepared
        target_cols (list str) : the name(s) of the dependent variable(s)
        test_size (float) : float between 0...1, the test set proportion

    Returns:
        X_train, X_test, y_train, y_test : python lists
    """

    data = data_to_list(data)
    data = order_sequences_by_length(data)

    X = [seq.drop(target_cols, axis=1) for seq in data]
    y = [pd.DataFrame(seq[target_cols]) for seq in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    data = (X_train, X_test, y_train, y_test)

    with open('./data/data.pickle', 'wb') as f:
        pickle.dump(data, f)

    return X_train, X_test, y_train, y_test

