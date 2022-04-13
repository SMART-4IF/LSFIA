from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import datacollection as datacollection

# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

label_map = {label: num for num, label in enumerate(datacollection.actions)}

model = Sequential()

sequences, labels = [], []


def start_model():
    load_seq()
    training_data = data_preparation()
    build_model()
    train_model(X_train=training_data.X_train, y_train=training_data.y_train)


class TrainingData:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def data_preparation():
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return TrainingData(X_train, X_test, y_train, y_test)


def build_model():
    # time steps = sequence_length - dimension = number of points per sequence
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(datacollection.actions.shape[0], activation='softmax'))


def train_model(X_train, y_train):
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=500)
    model.summary()


def load_seq():
    for action in datacollection.actions:
        print("Loading sequences for action = " + action)
        for sequence in np.array(os.listdir(os.path.join(datacollection.DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(datacollection.sequence_length):
                res = np.load(os.path.join(datacollection.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    print('Sequences = ' + str(sequences))
    print('Labels = ' + str(labels))