import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from pysleep.data_generator import create_dataset, prepare_train_test, DataGenerator
from pysleep.models.models import cnn_v2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
from datetime import datetime


# file_path = str(Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\saved_models\cnn_v2.h5'))
# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=False)#mode='max'
# early = EarlyStopping(monitor="acc", mode="max", patience=20, verbose=1)
# redonplat = ReduceLROnPlateau(monitor="acc", mode="max", patience=10, verbose=2)
# # callbacks_list = [checkpoint, early, redonplat]  # early
# log_dir = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\logs\fit')
# log_path = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
# # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
# callbacks = [tensorboard_callback,
#              checkpoint,
#              early,
#              redonplat]

model = cnn_v2(lr=0.005)
train, val, test = prepare_train_test()
X, y = create_dataset(train[0:20])
val_X, val_y = create_dataset(val[0:10])
hist = model.fit(X, y, validation_data=(val_X, val_y), batch_size=128, epochs=100）#, callbacks=callbacks)
print(hist.history)