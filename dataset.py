import numpy as np
import pandas_datareader.data as data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_batches_x(array, batch_size):
    q, mod = divmod(len(array), batch_size)
    array = array[:-mod]
    array = array.reshape((q, batch_size, 5, 1))
    return array


def create_batches_y(array, batch_size):
    q, mod = divmod(len(array), batch_size)
    array = array[:-mod]
    array = array.reshape((q, batch_size, 1))
    return array


def get_array():
    df = data.DataReader("AAPL", "yahoo", start="1/1/2010", end="1/1/2020")
    df = df[["High", "Low", "Open", "Close", "Volume"]]
    df = df.reset_index()
    df["Average"] = (df["High"] + df["Low"]) / 2

    scaler = MinMaxScaler()
    avg = np.array(df["Average"]).reshape(-1, 1)
    df["Average"] = scaler.fit_transform(avg).reshape(-1)

    x = []
    y = []

    for i in range(0, len(df["Average"]) - 5, 5):
        x.append(df.loc[i:i + 4, "Average"].tolist())
        y.append(df.loc[i + 5, "Average"].tolist())

    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    x = create_batches_x(x, 32)
    y = create_batches_y(y, 32)
    return x, y

