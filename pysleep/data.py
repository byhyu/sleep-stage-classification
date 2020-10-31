import functools

import numpy as np
from tensorflow.keras.utils import Sequence


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
        return X[in_file_idx:(in_file_idx + self.batch_size)], y[in_file_idx:(in_file_idx + self.batch_size)]
