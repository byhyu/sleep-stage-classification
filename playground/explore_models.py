#%%
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import functools
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, activations, layers, losses

import tensorflow as t
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
                mapping[(count, count+size)] = file_path
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
        # num_batches_per_file =
        # return math.ceil(len(self.x) / self.batch_size)

    @functools.lru_cache(maxsize=128)
    def _read_file(self, file_path):
        with np.load(file_path) as d:
            return d['x'],d['y']

    def __getitem__(self, idx):
        in_file_idx, file_path = self._find_file_path(idx)
        X, y = self._read_file(file_path)
        return X[in_file_idx:(in_file_idx+self.batch_size)], y[in_file_idx:(in_file_idx+self.batch_size)]


data_dir = Path(r'./data/physionet_sleep/eeg_fpz_cz')
data_files = list(data_dir.glob('*.npz'))

train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

data = np.load(data_files[0])
X = data['x']
y = data['y']

def create_baseline_cnn_model():
    model = Sequential()
    # # model.add(layers.Input(shape=(3000,1)))
    # # model.add(layers.Flatten(input_shape=(3000,1)))
    # model.add(layers.Convolution1D(16,kernel_size=5, activation=activations.relu, padding="valid", input_shape=(3000,1)))
    # # model.add(layers.Convolution1D(16,kernel_size=5, activation=activations.relu, padding="valid"))
    # model.add(layers.MaxPool1D(pool_size=2))
    # model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    # # model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    # model.add(layers.MaxPool1D(pool_size=2))
    # model.add(layers.Flatten())
    model.add(layers.Flatten(input_shape=(3000, 1)))
    # model.add(Dense(128,activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    model.summary()
    return model

def create_cnn_model2():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(3000, 1)))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    model.summary()
    return model
#%%
model = create_baseline_cnn_model()
data_generator = DataGenerator(data_files[0:100], batch_size=2)
model.fit(data_generator, verbose=2)
# model.fit(X, y, batch_size=2, verbose=2)
print('done')