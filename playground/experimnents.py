import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import datetime
from pysleep import models
from pysleep.models import create_baseline_dnn_model
from pysleep import models as SleepModels
import numpy as np
from pysleep.data_generator import DataGenerator, SeqDataGenerator
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tqdm import tqdm

data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz')
data_files = list(data_dir.glob('*.npz'))
print(f'number of data files: {len(data_files)}')

train_val, test = train_test_split(data_files, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

# model = models.cnn_v1()

# model1 = create_baseline_dnn_model()
model2 = SleepModels.create_cnn_model1()
# # model3 = SleepModels.create_dnn_model2()
model_cnn_cnn = SleepModels.create_cnn_cnn()

file_path = str(Path(r'..\saved_models\cnn_model1.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early
log_dir = Path(r'logs\fit')
log_path = log_dir/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
callbacks=[tensorboard_callback,
           checkpoint,
           early,
           redonplat]
# data_generator = DataGenerator(train[0:], batch_size=32)
seq_generator = SeqDataGenerator(train[0:], batch_size=4)
valid_generator = SeqDataGenerator(val[0:], batch_size=4)
X = np.random.random((200,10,3000,1))
y = np.random.random_integers(0,5,(200,10,1))
model_cnn_cnn.fit(X,y,
                  verbose=2,
                  epochs=3,
                  callbacks=callbacks)
# model_cnn_cnn.fit(seq_generator,
#                   verbose=2,
#                   epochs=100,
#                   callbacks=callbacks) #  steps_per_epoch=1000,    validation_steps=300,


# model2.fit(data_generator, verbose=2, callbacks=[tensorboard_callback,
#                                                  checkpoint,
#                                                  early,
#                                                  redonplat])#, callbacks=callbacks_list)
# model_cnn_cnn.fit(data_generator, verbose=2, callbacks=[tensorboard_callback])#, callbacks=callbacks_list)
#
# model.load_weights(file_path)
# preds = []
# gt = []
#
# evaluation
#%% test
# TODO: test model

print('done')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report

def convert_to_onehot(array):
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]

def plot_roc_auc(y_preds, y_tests, model_name=""):
    plt.figure()
    y_preds_oh = convert_to_onehot(y_preds)
    y_tests_oh = convert_to_onehot(y_tests)
    print(y_preds_oh)
    colors = ['', 'aqua', 'darkorange', 'cornflowerblue','', 'deeppink']
    labels = ['', 'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', '', 'Sleep stage R']
    for c in [1, 2, 3, 5]:
        fpr, tpr, _ = roc_curve(y_tests_oh[:, c], y_preds_oh[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC of class {labels[c]} = {roc_auc:0.2f})", color = colors[c])
    plt.legend(loc="lower right")
#     plt.show()
    plt.savefig("./plots/" + model_name + "_auc.png")