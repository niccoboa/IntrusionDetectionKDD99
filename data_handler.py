import numpy as np
import pandas as pd

from data_info import get_column_names
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def load(data):
    data = pd.read_csv(data, header=None)
    data.columns = get_column_names()
    return data


def gathering(data, label, attack_classes):
    data[label] = data[label].replace(attack_classes)
    return data


def preprocess(data):
    scaler = MinMaxScaler()  # scaler for numeric columns: scales each feature to a given range (default: [0, 1])
    le = LabelEncoder()  # label encoder for symbolic columns: encodes categorical into numerical (0, 1, 2, ...) values

    # SCALE NUMERIC COLUMNS
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()[:-1]  # numeric columns (& exclude target)
    scaler.fit(data[numeric_columns])  # fit scaler on training data
    data[numeric_columns] = scaler.transform(data[numeric_columns])  # transform data using the scaler

    # ENCODE SYMBOLIC COLUMNS
    symbolic_columns = ['protocol_type', 'service', 'flag']  # symbolic columns (excluding target)
    for col in symbolic_columns:
        data[col] = le.fit_transform(data[col])  # fit and transform symbolic columns

    # ENCODE TARGET COLUMN
    target = data['attack_type']  # select target column
    data['attack_type'] = le.fit_transform(target)   # fit and transform target column

    return data
