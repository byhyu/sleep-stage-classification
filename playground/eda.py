#%%
import numpy as np
from pathlib import Path
import streamlit as st


data_dir = Path(r'./data/physionet_sleep/eeg_fpz_cz')
data_files = list(data_dir.glob('*.npz'))
data_file = st.sidebar.selectbox('data file', data_files)

def load_data(data_file):
    data = np.load(data_file)
    return data

#%%
data = load_data(data_files[0])
