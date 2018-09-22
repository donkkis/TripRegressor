import unittest
import pandas as pd
import numpy as np
import random
from preprocess_utils import *
from prepare_data import *


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
        data1.append(df.loc[0:2, :])
        data1.append(df.loc[3:4, :])
        data1.append(df.loc[5:8, :])
        data1.append(pd.DataFrame(df.loc[9, :]).transpose())

        data2 = data_to_list(df, 'col2')
        self.assertEqual(len(data2), 4)
        self.assertTrue(data1[0].equals(data2[0]))
        self.assertTrue(data1[1].equals(data2[1]))
        self.assertTrue(data1[2].equals(data2[2]))
        self.assertTrue(data1[3].equals(data2[3]))

    def test_order_sequences_by_length(self):
        d = {'col1' : random.sample(range(0, 100), 10), 'col2' : [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]}
        df = pd.DataFrame(data=d)

        data = []
        data.append(df.loc[0:2, :])
        data.append(df.loc[3:4, :])
        data.append(df.loc[5:8, :])
        data.append(pd.DataFrame(df.loc[9, :]).transpose())

        data = order_sequences_by_length(data)
        self.assertEqual(max(data[0]['col2']), 4)
        self.assertEqual(max(data[1]['col2']), 2)
        self.assertEqual(max(data[2]['col2']), 1)
        self.assertEqual(max(data[3]['col2']), 3)

    def test_dataset_to_numpy(self):
        pass

    def test_prepare_dataset




if __name__ == "__main__":
    unittest.main()
