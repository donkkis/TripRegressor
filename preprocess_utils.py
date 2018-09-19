import pandas as pd
import matplotlib.pyplot as plt
import warnings
from math import radians, cos, sin, asin, sqrt
from typing import List, Any


class BoundingBox(object):

    def __init__(self, lat_min, lat_max, long_min, long_max):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.long_min = long_min
        self.long_max = long_max

    def contains(self, lat, long):
        return self.lat_min <= lat <= self.lat_max and self.long_min <= long <= self.long_max


class BoundingBoxGrid(object):

    boxes: List[Any]

    def __init__(self, boxes=[]):
        self.boxes = boxes

    def init_from_histogram(self, h, xedges, yedges, tolerance):
        """
        consumes h, xedged, yedges variables provided by matplotlib.pyplot.hist2d
        and constructs the bounding boxes according to the specified tolerance
        """

        if len(self.boxes) > 0:
            warnings.warn("BoundingBoxSet has been previously initialized, setting self.boxes = []")
            self.boxes = []

        rows = h.shape[0]
        cols = h.shape[1]

        for row in range(rows):
            for col in range(cols):
                if h[row][col] > tolerance:
                    long_min = xedges[row]
                    long_max = xedges[row + 1]
                    lat_min = yedges[col]
                    lat_max = yedges[col + 1]

                    self.boxes.append(BoundingBox(lat_min, lat_max, long_min, long_max))


def check_location_against_bbox_set(row, bbox_set):
    """
    Check if a provided location is contained in any of the boundingboxes

    Args:
        row (pd.DataFrame) : a (1, n_features) shaped slice of a pandas.DataFrame instance
        bbox_set (BoundingBoxGrid) : contains the bounding boxes each row in dataset should be checked against

    Returns:
        True / False depending if the provided location is contained in any of the boundingboxes
    """

    for box in bbox_set.boxes:
        if box.contains(row['gpslat'], row['gpslong']):
            return True
    return False


def check_tolerances_dataset(data, bbox_set):
    """
    Apply check_location_against_bbox_set for each row in dataset

    Args:
        data (pd.DataFrame) : must contain numerical columns 'gpslat' and 'gpslong'
        bbox_set (BoundingBoxGrid) :

    Returns:
        data (pd.DataFrame) : The transformed DataFrame, with a new column indicating the result of
            check_location_against_bbox_set for each row
    """
    data.loc[:, 'is_in_tolerance_area'] = data.apply(check_location_against_bbox_set, args=[bbox_set], axis=1)
    return data


def build_sample_indices(data, col):
    """
    Builds sequence indices based on running data (e.g. duration in seconds)
    Iterate over the rows in data and increment index whenever data[col] == 0

    Args:
        data (pd.DataFrame) : The dataframe whose indices are to be built.
        col (string) : The column of the dataframe where the indexable variable (eg. running duration) lives
    Returns:
         data (pd.DataFrame) : Data with an added column for the sequence indices
    """

    idx = 0
    seq_indices = []
    for _, row in data.iterrows():
        if row[col] == 0:
            # increment index and begin new sequence
            idx += 1
            seq_indices.append(idx)
        else:
            seq_indices.append(idx)

    data['sequence'] = seq_indices

    return data


def drop_sequences(data, cond_col, condition, threshold):
    """
    Drop sequences including > threshold rows where the column
    cond_col evaluates to condition

    Args:
        data (pd.DataFrame) : data to be checked
        cond_col (string) : the name of the column to be checked for a logical condition
        condition (bool) : The condition
        threshold (int) : occurence threshold for removing the sequence
    Returns:
        data_dropped (pd.DataFrame) : data where the affected rows have been dropped
    """
    seq_ids = list(data['sequence'].unique())
    to_be_dropped = []
    for seq_id in seq_ids:
        seq = data[data['sequence'] == seq_id]
        try:
            if seq[cond_col].value_counts()[condition] > threshold:
                to_be_dropped.append(seq.index.tolist())
        except:
            pass

    to_be_dropped = [idx for seq in to_be_dropped for idx in seq]
    data_dropped = data.drop(to_be_dropped)
    return data_dropped


def haversine(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def compute_distance_between_rows(row1, row2, gpslat_col, gpslong_col):
    """
    Compute the haversine distance given two rows sliced from a pd.DataFrame.
    Rows are expected to be identical in their columns

    Args:
        row1 (pd.Series of shape (1, n_features))
        row2 (pd.Series of shape (1, n_features))
        gpslat_col (string) : the identifier of the column of gps latitude in both rows
        gpslong_col (string) : the identifier of the column of gps longitude in both rows

    Returns:
        distance (float)
    """
    lon1 = row1[gpslong_col]
    lat1 = row1[gpslat_col]
    lon2 = row2[gpslong_col]
    lat2 = row2[gpslat_col]

    return haversine(lon1, lat1, lon2, lat2)


def convert_sequence_to_relative(seq):
    """
    Converts gpslat, gpslong and altitude to running relative metrics

    Args:
        seq (pd.DataFrame) : the sequence to be converted

    Returns:
        distance (pd.Series) : running distance offset (meters) from the beginning of the series
        rel_altitude (pd.Series) : running altitude offset (meters) from the beginning of the series
        rel_soc (pd.Series) : the additive inverse of SOC offset from beginning of the series
    """
    first_row = seq.iloc[0, :]
    args = (first_row, 'gpslat', 'gpslong')

    distance = seq.apply(compute_distance_between_rows, args=args, axis=1)
    rel_altitude = seq.apply(lambda row: row['altitude'] - first_row['altitude'], axis=1)
    rel_soc = seq.apply(lambda row: -1 * (row['soc'] - first_row['soc']), axis=1)

    return distance, rel_altitude, rel_soc


def convert_dataset_to_relative(data):
    """
    Iterate over sequences in the whole dataset and make the necessary conversions

    Args:
        data (pandas.DataFrame) : Must contain numerical columns 'sequence', 'gpslat', 'gpslong', 'altitude', 'soc'

    Returns:
        data (pandas.DataFrame) : Transformed version of the dataset where each row has been applied the transformations
            described in convert_sequence_to_relative
    """
    seq_ids = list(data['sequence'].unique())

    for seq_id in seq_ids:
        seq = data[data['sequence'] == seq_id]
        distance, rel_altitude, rel_soc = convert_sequence_to_relative(seq)
        data.loc[seq.index, 'distance'] = distance
        data.loc[seq.index, 'rel_altitude'] = rel_altitude
        data.loc[seq.index, 'rel_soc'] = rel_soc

    data = data.drop(['gpslat', 'gpslong', 'altitude', 'soc'], axis=1)

    return data


def preprocess_dataset(data, bins, threshold):
    """
    Preprocess a multivariate timeseries dataset from a BEV

    Args:
        data (pandas.DataFrame) : a pandas.DataFrame instance with string column 'timestamp' and
            numericals gpslong, gpslat, altitude, soc, temp, duration
        bins (int) : Number of bins to be used in creation of the 2d histogram to be fed forward to BoundingBoxGrid.
            The resulting histogram will be of dimension bins x bins
        threshold: Occurence threshold of non-compliant rows for removing a sequence

    Returns:
        data (pandas.DataFrame) : the processed data with numerical columns speed, temp, duration, sequence,
            reL_altitude, rel_soc

    """

    h, xedges, yedges, _ = plt.hist2d(data['gpslong'], data['gpslat'], bins=bins)
    bbox_set = BoundingBoxGrid()
    bbox_set.init_from_histogram(h, xedges, yedges, 1)

    # Apply transformations
    data = check_tolerances_dataset(data, bbox_set)
    data = build_sample_indices(data, 'duration')
    data = drop_sequences(data, 'is_in_tolerance_area', False, threshold=threshold)
    data = convert_dataset_to_relative(data)

    data = data.drop('is_in_tolerance_area', axis=1)
    data = data.drop('timestamp', axis=1)

    return data
