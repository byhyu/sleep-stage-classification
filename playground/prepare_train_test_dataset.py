from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz')

data_files = list(data_dir.glob('*.npz'))

train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

train_dataset = {k:np.load(k) for k in train}
test_dataset = {k:np.load(k) for k in test}
val_dataset = {k:np.load(k) for k in val}