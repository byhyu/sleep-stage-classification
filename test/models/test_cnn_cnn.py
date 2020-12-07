import pytest
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
from pathlib import Path
from pysleep.data_generator import DataGenerator, SeqDataGenerator, SeqOneOutGenerateor
from pathlib import Path
from sklearn.model_selection import train_test_split
from pysleep.models import cnn_cnn, cnn_cnn_1, cnn_v0
from pysleep.data_generator import create_dataset

file_path = str(Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\saved_models\cnn_v0.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=False)#mode='max'
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early
log_dir = Path(r'logs\fit')
log_path = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
callbacks = [tensorboard_callback,
             checkpoint,
             early,
             redonplat]



def prepare_Xy(X_dims=(100,10,3000,1), y_dims =(100,1)):
    X = np.random.random(X_dims)
    y = np.random.randint(0,5,y_dims)
    return X, y

def prepare_train_test():
    data_dir = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\data\physionet_sleep\eeg_fpz_cz')
    data_files = list(data_dir.glob('*.npz'))
    print(f'number of data files: {len(data_files)}')

    train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    return train, val, test

def prepare_Xy_from_gen():
    # data_dir = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\data\physionet_sleep\eeg_fpz_cz')
    # data_files = list(data_dir.glob('*.npz'))
    # print(f'number of data files: {len(data_files)}')
    #
    # train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
    # train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    train, val, test = prepare_train_test()
    Xy = SeqDataGenerator(train[0:10], batch_size=4)
    # y = SeqDataGenerator(val[0:3], batch_size=4)
    return Xy

def test_cnn_v0():
    train, val, test = prepare_train_test()
    Xy = DataGenerator(train[0:3], batch_size=4)
    val_gen = DataGenerator(val[0:2],batch_size=4)
    model = cnn_v0()

    # file_path = str(Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\saved_models\cnn_v0.h5'))
    # checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
    # redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=10, verbose=2)
    # # callbacks_list = [checkpoint, early, redonplat]  # early
    # log_dir = Path(r'logs\fit')
    # log_path = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
    # callbacks = [tensorboard_callback,
    #              checkpoint,
    #              early,
    #              redonplat]

    hist = model.fit(Xy, validation_data=val_gen, epochs=50, callbacks=callbacks)
    print(hist.history)


def test_seq_gen():
    X, y = prepare_Xy_from_gen()

def test_seq_one_out_gen():
    # data_dir = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\data\physionet_sleep\eeg_fpz_cz')
    # data_files = list(data_dir.glob('*.npz'))
    # print(f'number of data files: {len(data_files)}')
    #
    # train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
    # train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    train, val, test = prepare_train_test()

    Xy = SeqOneOutGenerateor(train[0:3], batch_size=4)

def test_cnn_cnn_1_arr_input():
    X, y = prepare_Xy()
    model = cnn_cnn_1()
    hist = model.fit(X,y, epochs=5)
    print(hist.history)

def test_cnn_cnn_1_seq_input():
    train, val, test = prepare_train_test()
    Xy = SeqOneOutGenerateor(train[0:], batch_size=4)
    model = cnn_cnn_1()


    hist = model.fit(Xy, epochs=20, callbacks=callbacks)
    print(hist.history)

def test_cnn_cnn_seq_input():
    model = cnn_cnn()
    Xy = prepare_Xy_from_gen()
    # print(y)
    hist = model.fit(Xy, epochs=5)
    print(hist.history)


def test_create_dataset():
    train, val, test = prepare_train_test()
    X, y = create_dataset(train[0:2])

def test_cnn_v01():
    from pysleep.models import cnn_v01
    model = cnn_v01()
    train, val, test = prepare_train_test()
    X,y = create_dataset(train[0:])
    val_X, val_y = create_dataset(val[0:3])
    hist = model.fit(X, y, batch_size=4, epochs=10, callbacks=callbacks)
    print(hist.history)

def test_cnn_v1():
    from pysleep.models import cnn_v1
    model = cnn_v1()
    train, val, test = prepare_train_test()
    X,y = create_dataset(train[0:])
    val_X, val_y = create_dataset(val[0:3])
    hist = model.fit(X, y, validation_data=(val_X,val_y), batch_size=4, epochs=10, callbacks=callbacks)
    print(hist.history)


    # Xy = DataGenerator(train[0:3], batch_size=4)
    # val_gen = DataGenerator(val[0:2], batch_size=4)
    #
    # hist = model.fit(Xy, validation_data=val_gen, epochs=50, callbacks=callbacks)
    # print(hist.history)

#
# train, val, test = prepare_train_test()
# Xy = DataGenerator(train[0:3], batch_size=4)
# val_gen = DataGenerator(val[0:2], batch_size=4)
# model = cnn_v0()
# hist = model.fit(Xy, validation_data=val_gen, epochs=50, callbacks=callbacks)
# print(hist.history)
