import unittest
import pandas as pd
import numpy as np
import random
import uuid
from preprocess_utils import *
import prepare_data as prep


# ---STATIC---

def get_random_dataframe() -> (pd.DataFrame, list, list):
    """
    Helper for getting DataFrames with random dimensions/data and
    populated with a dummy sequence id column for testing.

    Returns:
        df (pandas.Dataframe)

    """
    n_rows = np.random.randint(2, 50)
    n_target_cols = np.random.randint(2, 50)
    n_predictor_cols = np.random.randint(2, 50)

    # Generate random data, +1 for the indexing variable
    rand_array = np.around(np.random.randn(n_rows, n_target_cols + n_predictor_cols + 1), decimals=2)

    # Assign random unique names to columns, idx is the indexing variable
    # Some tests might fail if column names are not unique!
    predictor_cols = [uuid.uuid4() for i in range(n_predictor_cols)] + ['idx']
    target_cols = [uuid.uuid4() for i in range(n_target_cols)]
    cols = predictor_cols + target_cols

    df = pd.DataFrame(data=rand_array, columns=cols)

    # Build dummy sequence indexing
    idx = []
    run_index = np.random.randint(0, 10)
    for i in range(n_rows):
        coinflip = np.random.random()
        if coinflip >= 0.75:
            run_index = run_index + 1
        idx.append(run_index)
    df['idx'] = idx

    return df, predictor_cols, target_cols


# ---TESTS---

class TestPreprocessFunctions(unittest.TestCase):

    def test_check_location_against_bbox_set(self):
        pass

    def test_check_tolerances_dataset(self):
        pass

    def test_build_sample_indices(self):
        pass

    def test_drop_sequences(self):
        pass

    def test_haversine(self):
        pass

    def test_compute_distance_between_rows(self):
        pass

    def test_convert_sequence_to_relative(self):
        pass

    def test_convert_dataset_to_relative(self):
        pass

    def test_preprocess_dataset(self):
        pass


class TestPrepareData(unittest.TestCase):

    def test_data_to_list(self):
        d = {'col1': random.sample(range(0, 100), 10), 'col2': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]}
        df = pd.DataFrame(data=d)

        data1 = []
        data1.append(pd.DataFrame(df.loc[0:2, :]))
        data1.append(pd.DataFrame(df.loc[3:4, :]))
        data1.append(pd.DataFrame(df.loc[5:8, :]))
        data1.append(pd.DataFrame(df.loc[9:9, :]))

        prep.SEQID_COL = 'col2'
        data2 = prep.data_to_list(df)

        self.assertEqual(len(data2), 4)
        self.assertTrue(data1[0].equals(data2[0]))
        self.assertTrue(data1[1].equals(data2[1]))
        self.assertTrue(data1[2].equals(data2[2]))
        self.assertTrue(data1[3].equals(data2[3]))

    def test_order_sequences_by_length(self):
        d = {'col1': random.sample(range(0, 100), 10), 'col2': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]}
        df = pd.DataFrame(data=d)

        data_ordered = []
        data_ordered.append(pd.DataFrame(df.loc[9, :]).transpose())
        data_ordered.append(df.loc[3:4, :])
        data_ordered.append(df.loc[0:2, :])
        data_ordered.append(df.loc[5:8, :])

        data1 = []
        data1.append(df.loc[0:2, :])
        data1.append(df.loc[3:4, :])
        data1.append(df.loc[5:8, :])
        data1.append(pd.DataFrame(df.loc[9, :]).transpose())

        data1 = prep.order_sequences_by_length(data1)
        self.assertTrue(data1[0].equals(data_ordered[0]))
        self.assertTrue(data1[1].equals(data_ordered[1]))
        self.assertTrue(data1[2].equals(data_ordered[2]))
        self.assertTrue(data1[3].equals(data_ordered[3]))

    def test_dataset_to_numpy(self):
        X = []
        y = []
        randX = []
        randy = []

        for i in range(5):
            rows = np.random.randint(2, 50)
            cols = np.random.randint(2, 50)

            randX.append(np.around(np.random.randn(rows, cols), decimals=2))
            randy.append(np.around(np.random.randn(rows, cols), decimals=2))

            X.append(pd.DataFrame(randX[i]))
            y.append(pd.DataFrame(randy[i]))

            randX[i] = np.reshape(randX[i], newshape=(1, randX[i].shape[0], randX[i].shape[1]))
            randy[i] = np.reshape(randy[i], newshape=(1, randy[i].shape[0], randy[i].shape[1]))

        X_numpy, y_numpy = prep.dataset_to_numpy(X, y)

        for i in range(5):
            np.testing.assert_array_equal(randX[i], X_numpy[i])
            np.testing.assert_array_equal(randy[i], y_numpy[i])

    def test_prepare_dataset_with_no_excludes_no_ordering_no_normalize(self):
        prep.SEQID_COL = 'idx'
        df, predictor_cols, target_cols = get_random_dataframe()

        X, y = prep.prepare_dataset(df, target_cols, order=False, normalize_data=False, test_size=0.2)

        seq_ids = list(df['idx'].unique())

        sample: np.ndarray
        for seq_id, sample in zip(seq_ids, X):
            expect_steps = len(df[df['idx'] == seq_id])
            expect_features = len(predictor_cols)
            expect_dataframe = df.loc[df['idx'] == seq_id, predictor_cols]
            expect_matrix = np.array(expect_dataframe, ndmin=3)

            self.assertIsInstance(sample, np.ndarray)
            self.assertTrue(sample.ndim == 3)
            self.assertTrue(sample.shape == (1, expect_steps, expect_features))
            np.testing.assert_array_equal(expect_matrix, sample)

    def test_prepare_dataset_with_excludes(self):
        prep.SEQID_COL = 'idx'
        df, predictor_cols, target_cols = get_random_dataframe()

        exclude_cols = [predictor_cols[0]]

        X, y = prep.prepare_dataset(df, target_cols, exclude=exclude_cols,
                                    order=False, normalize_data=False, test_size=0.2)

        seq_ids = list(df['idx'].unique())

        for seq_id, sample in zip(seq_ids, X):
            expect_steps = len(df[df['idx'] == seq_id])
            expect_features = len(predictor_cols) - len(exclude_cols)
            expect_dataframe = df.loc[df['idx'] == seq_id, predictor_cols[1:]]
            expect_matrix = np.array(expect_dataframe, ndmin=3)

            self.assertIsInstance(sample, np.ndarray)
            self.assertTrue(sample.ndim == 3)
            self.assertTrue(sample.shape == (1, expect_steps, expect_features))
            np.testing.assert_array_equal(expect_matrix, sample)

    def test_prepare_dataset_with_ordering(self):
        prep.SEQID_COL = 'seq_id'
        df1 = pd.read_csv('./data/dummy_ordered.csv')
        X1, y1 = prep.prepare_dataset(df1, target_cols=['soc'], order=False, normalize_data=False)

        df2 = pd.read_csv('./data/dummy_unordered.csv')
        X2, y2 = prep.prepare_dataset(df2, target_cols=['soc'], order=True, normalize_data=False)

        self.assertTrue(len(X1) == len(X2) == len(y1) == len(y2))

        for example1, label1, example2, label2 in zip(X1, y1, X2, y2):
            np.testing.assert_array_equal(example1, example2)
            np.testing.assert_array_equal(label1, label2)

    def test_prepare_dataset_with_normalization(self):
        prep.SEQID_COL = 'seq_id'
        df1 = pd.read_csv('./data/dummy_ordered.csv')
        X1, y1 = prep.prepare_dataset(df1, target_cols=['soc'], order=False, normalize_data=True)

        df2 = pd.read_csv('./data/dummy_standardized.csv')
        X2, y2 = prep.prepare_dataset(df2, target_cols=['soc'], order=False, normalize_data=False)

        self.assertTrue(len(X1) == len(X2) == len(y1) == len(y2))

        for example1, label1, example2, label2 in zip(X1, y1, X2, y2):
            np.testing.assert_array_almost_equal(example1, example2)
            np.testing.assert_array_almost_equal(label1, label2)

    def test_prepare_dataset_with_normalization_and_ordering(self):
        prep.SEQID_COL = 'seq_id'
        df1 = pd.read_csv('./data/dummy_unordered.csv')
        X1, y1 = prep.prepare_dataset(df1, target_cols=['soc'], order=True, normalize_data=True)

        df2 = pd.read_csv('./data/dummy_standardized.csv')
        X2, y2 = prep.prepare_dataset(df2, target_cols=['soc'], order=False, normalize_data=False)

        self.assertTrue(len(X1) == len(X2) == len(y1) == len(y2))

        for example1, label1, example2, label2 in zip(X1, y1, X2, y2):
            np.testing.assert_array_almost_equal(example1, example2)
            np.testing.assert_array_almost_equal(label1, label2)

    def test_prepare_dataset_with_normalization_ordering_and_excludes(self):
        prep.SEQID_COL = 'seq_id'
        df1 = pd.read_csv('./data/dummy_unordered.csv')
        df1 = df1.drop(['timestamp', 'gpslat'], axis=1)
        X1, y1 = prep.prepare_dataset(df1, target_cols=['soc'], order=True, normalize_data=True)

        df2 = pd.read_csv('./data/dummy_standardized.csv')
        X2, y2 = prep.prepare_dataset(df2, target_cols=['soc'],
                                      exclude=['timestamp', 'gpslat'], order=False, normalize_data=False)

        self.assertTrue(len(X1) == len(X2) == len(y1) == len(y2))

        for example1, label1, example2, label2 in zip(X1, y1, X2, y2):
            np.testing.assert_array_almost_equal(example1, example2)
            np.testing.assert_array_almost_equal(label1, label2)

    def test_normalize(self):

        col1 = random.sample(range(0, 100), 10)
        col2 = random.sample(range(0, 100), 10)
        col4 = random.sample(range(0, 100), 10)

        d = {'col1': col1,
             'col2': col2,
             'col3': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
             'col4': col4}
        df = pd.DataFrame(data=d)

        prep.SEQID_COL = 'col3'

        norm_data = prep.normalize(df.copy())
        self.assertFalse(df['col1'].equals(norm_data['col1']))
        self.assertFalse(df['col2'].equals(norm_data['col2']))
        self.assertTrue(df['col3'].equals(norm_data['col3']))
        self.assertFalse(df['col4'].equals(norm_data['col4']))


if __name__ == "__main__":
    unittest.main()
