# from models.cnn import CNN1Head
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, activations, layers, losses
from tensorflow.keras.layers import *
from models.dnn import DNN

from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
from ..data_loader import create_dataset, prepare_train_test
from ..data_loader import DataGenerator
model = DNN(model_name='dnn_model')
model = model.build_model()
print('test')