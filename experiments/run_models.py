import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from pathlib import Path
from pysleep.data.data_generator import DataGenerator, SeqOneOutGenerateor, SeqDataGenerator
from pysleep.data.data_generator import create_dataset, prepare_train_test
from pysleep.models.cnn_models import CNN1Head
# prepare data
data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz').resolve()
print(f"data dir:{data_dir}")
train, val, test = prepare_train_test(data_dir=data_dir)

X, y = create_dataset(train[0:5])
val_X, val_y = create_dataset(val[0:2])

# train model
model = CNN1Head()
model = model.build_model()
hist = model.fit(X, y, validation_data=(val_X, val_y), batch_size=32, epochs=10) #, callbacks=callbacks)
print(hist.history)
# test model

# plot

