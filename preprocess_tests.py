import unittest
import pandas as pd
import numpy as np
import random
from preprocess_utils import *
import prepare_data as prep


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
        d = {'col1' : random.sample(range(0, 100), 10), 'col2' : [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]}
        df = pd.DataFrame(data=d)

        data1 = []
        data1.append(pd.DataFrame(df.loc[0:2, 'col1']))
        data1.append(pd.DataFrame(df.loc[3:4, 'col1']))
        data1.append(pd.DataFrame(df.loc[5:8, 'col1']))
        data1.append(pd.DataFrame(df.loc[9:9, 'col1'].transpose()))

        prep.SEQID_COL = 'col2'

        data2 = prep.data_to_list(df)
        self.assertEqual(len(data2), 4)
        self.assertTrue(data1[0].equals(data2[0]))
        self.assertTrue(data1[1].equals(data2[1]))
        self.assertTrue(data1[2].equals(data2[2]))
        self.assertTrue(data1[3].equals(data2[3]))

    def test_order_sequences_by_length(self):
        d = {'col1' : random.sample(range(0, 100), 10), 'col2' : [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]}
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
        pass

    def test_prepare_dataset(self):
        pass

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
