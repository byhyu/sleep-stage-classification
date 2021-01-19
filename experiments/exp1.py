from pathlib import Path
import functools
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import *

data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz').resolve()
def prepare_train_test(data_dir = data_dir):
    print('start')
    data_files = list(data_dir.glob('*.npz'))
    print(f'number of data files: {len(data_files)}')

    train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    return train, val, test
def create_dataset(data_files):
    Xs = np.empty([0,3000,1])
    ys = np.empty([0,])
    for dfile in data_files:
        with np.load(dfile) as d:
            x=d['x']
            y=d['y']
            print(f'x shape:{x.shape}')
            print(f'y shape:{y.shape}')
            Xs = np.vstack((Xs, x))
            ys = np.hstack((ys,y))
    return Xs, ys

class DataGenerator(Sequence):
    def __init__(self, file_list, batch_size=32):
        self.file_list = file_list
        self.idx_file_mapping, self.num_samples = self._create_idx_file_mapping()
        self.batch_size = batch_size

    def _create_idx_file_mapping(self):
        count = 0
        mapping = {}
        for file_path in self.file_list:
            with np.load(file_path) as d:
                size = d['y'].shape[0]
                mapping[(count, count + size)] = file_path
            count += size
        return mapping, count

    def _find_file_path(self, idx):
        for idx_range, file_path in self.idx_file_mapping.items():
            start, end = idx_range
            if start <= idx and idx <= end:
                in_file_idx = idx - start
                return (in_file_idx, file_path)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    @functools.lru_cache(maxsize=128)
    def _read_file(self, file_path):
        with np.load(file_path) as d:
            return d['x'], d['y']

    def __getitem__(self, idx):
        in_file_idx, file_path = self._find_file_path(idx)
        X, y = self._read_file(file_path)
        # y = to_categorical(y,5)
        X_batch = X[in_file_idx:(in_file_idx + self.batch_size)]
        X_batch = rescale_array(X_batch)
        y_batch = y[in_file_idx:(in_file_idx + self.batch_size)]
        # y_batch = to_categorical(y_batch, 5)
        return X_batch, y_batch #,[None]

def cnn_v01(n_outputs=5):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_v1(lr=0.005):
    model = Sequential(layers=[
        layers.Convolution1D(32, kernel_size=5, strides=1, activation='relu', padding='valid', input_shape=(3000,1)),
        layers.Convolution1D(32, kernel_size=5, strides=1, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(64, kernel_size=25, strides=6, activation='relu', padding='valid', input_shape=(3000, 1)),
        layers.Convolution1D(64, kernel_size=50, strides=6, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.GlobalMaxPool1D(),
        layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.05),
        layers.Dense(5, activation='softmax')]
    )
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy']) #,class_model='categorical'
    return model

model = cnn_v01()
train, val, test = prepare_train_test(data_dir = data_dir)
trainx, trainy = create_dataset(train[0:1])
# train_dl = DataGenerator(train[0:3])
hist = model.fit(trainx, trainy, batch_size=4, epochs=10)
print(hist.history)