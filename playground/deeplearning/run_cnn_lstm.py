from pathlib import Path
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pysleep.data.data_generator import prepare_train_test, create_dataset, DataGenerator, SeqOneOutGenerateor
from pysleep.models.cnn_lstm import CNN_LSTM

data_dir = Path(r'../../data/physionet_sleep/eeg_fpz_cz').resolve()
train, val, test = prepare_train_test(data_dir=data_dir)
print(len(train))

file_path = str(Path(r'../saved_models/cnn_model1.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early
log_dir = Path(r'logs/fit')
log_path = log_dir/ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
callbacks=[tensorboard_callback,
           checkpoint,
           early,
           redonplat]

train_dl = SeqOneOutGenerateor(train)
val_dl = SeqOneOutGenerateor(val)
# model = CNN1Head(model_name='CNN1Head_train3_test2', epochs=20, learning_rate=0.005, batch_size=32)
model = CNN_LSTM(model_name='CNN_LSTM')
model.build_model()
hist = model.fit(train_dl, validatation_data=val_dl, epochs=50)
print(hist.history)

