import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


def get_current_date_time():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


def check_checkpoints_available(path="."):
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_train_data(file, eval_size=0.20):
    # Load and prepare
    df = pd.read_csv(file)
    df = df.dropna()
    x = df["Image"]
    y = df.drop(["Image"], axis=1)

    # Split train and eval set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=eval_size, random_state=42)

    # Convert and scale
    x_train = np.array([np.array(image.split(" ")).reshape(96, 96, 1) for image in x_train], dtype=float) / 255.0
    x_val = np.array([np.array(image.split(" ")).reshape(96, 96, 1) for image in x_val], dtype=float) / 255.0
    y_train = (y_train.values - 48) / 48
    y_val = (y_val.values - 48) / 48

    return x_train, x_val, y_train, y_val


def load_test_data(file):
    # Load and prepare
    df = pd.read_csv(file)
    df = df.dropna()
    x = df["Image"]

    # Convert and scale
    x = np.array([np.array(image.split(" ")).reshape(96, 96, 1) for image in x], dtype=float)

    return x


def get_batch(x, y=None, step=1):
    if y is None:
        data_length = len(x)
        for ndx in range(0, data_length, step):
            yield x[ndx:min(ndx + step, data_length)]
    else:
        assert len(x) == len(y)
        data_length = len(x)
        for ndx in range(0, data_length, step):
            yield x[ndx:min(ndx + step, data_length)], y[ndx:min(ndx + step, data_length)]


def add_keypoints_to_image(image, key_points):
    updated_image = np.copy(image)
    for key_point in key_points[0]:
        for x, y in zip(key_point[0], key_point[1]):
            cv2.circle(updated_image, (x, y), 5, (0, 0, 255), -1)
    return updated_image
