from pysleep import models
from pysleep.models import create_baseline_dnn_model
from pysleep import models as SleepModels

from pysleep.data import DataGenerator
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz')
data_files = list(data_dir.glob('*.npz'))
print(f'number of data files: {len(data_files)}')

train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

data_generator = DataGenerator(train, batch_size=32)
model1 = create_baseline_dnn_model()
model2 = SleepModels.create_cnn_model1()
model3 = SleepModels.create_dnn_model2()

file_path = str(Path(r'..\saved_models\dnn_model2.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model3.fit(data_generator, validation_data=DataGenerator(val), verbose=2, callbacks=callbacks_list)
#%% test
# TODO: test model

print('done')