import sys
sys.path.append(".")
sys.path.append("..")

from models.cnn import CNN1Head
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
from ..data_loader import create_dataset, prepare_train_test
from ..data_loader import DataGenerator
model = CNN1Head()
model = model.build_model()

file_path = str(Path(r'out\saved_models\cnn_v1.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=False)#mode='max'
early = EarlyStopping(monitor="acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="acc", mode="max", patience=10, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early
log_dir = Path(r'out\logs\fit')
log_path = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
callbacks = [tensorboard_callback,
             checkpoint,
             early,
             redonplat]


train, val, test = prepare_train_test()
# X, y = create_dataset(train[0:4])
# val_X, val_y = create_dataset(val[0:2])

train_gen = DataGenerator(train[0:4])
val_gen = DataGenerator(val[0:2])
hist = model.fit(train_gen, validation_data=val_gen, epochs=4, callbacks=callbacks)
print(hist.history)