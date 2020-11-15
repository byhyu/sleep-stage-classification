from pathlib import Path
import numpy as np
from pysleep.data import SeqDataGenerator
data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz')
data_files = list(data_dir.glob('*.npz'))
# for dfile in data_files:
#     with np.load(dfile) as data:
#         x = data['x']
#         y = data['y']
#         print(f'x shape:{x.shape}')
#         print(f'y shape:{y.shape}')
#         if y.shape == 0:
#             print(dfile)

#%% train
train = data_files[0:3]
seq_gen = SeqDataGenerator(train, batch_size=4, time_steps=10)
