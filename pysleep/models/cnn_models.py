import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import MaxPooling1D
# from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from pysleep.data_generator import create_dataset,prepare_train_test

from pysleep.models.base_model import BaseModel
from pathlib import Path

# saved_model_dir = Path(r'C:\Users\hyu\github-repos\LearnFromSleepData\saved_models')
saved_model_dir = Path(r'out\saved_models')
class CNN1Head(BaseModel):
	def build_model(self):
		n_timesteps = self.n_timesteps
		n_features = self.n_channels
		n_outputs =self.n_outputs
		model = Sequential(layers=[
			Input(shape=(n_timesteps, n_features)),
			Conv1D(filters=64, kernel_size=3, activation='relu'),
			Dropout(0.5),
			MaxPooling1D(pool_size=2),
			Flatten(),
			Dense(128, activation='relu'),
			Dense(n_outputs, activation='softmax')
		])
		model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(self.learning_rate), metrics=['sparse_categorical_accuracy'])
		self.model = model
		return model

	def fit(self, trainX, trainy, valX, valy):
		self.model.fit(trainX, trainy,
					   validation_split=0.4,
					  epochs=self.epochs,
					  batch_size=self.batch_size,
					  verbose=2,
					  callbacks=self.callbacks)
	def evaluate(self, testX, testy):
		_, accuracy = self.model.evaluate(testX, testy, batch_size=self.batch_size, verbose=2)
		return accuracy

	def plot_confusion_matrix(self, y_true, y_pred):
		pass

class CNN2Head(BaseModel):
	def build_model(self):
		n_timesteps = self.n_timesteps
		n_features = self.n_channels
		n_outputs = self.n_outputs

		# head 1
		inputs1 = Input(shape=(n_timesteps, n_features))
		conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
		drop1 = Dropout(0.5)(conv1)
		pool1 = MaxPooling1D(pool_size=2)(drop1)
		flat1 = Flatten()(pool1)
		# head 2
		inputs2 = Input(shape=(n_timesteps, n_features))
		conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
		drop2 = Dropout(0.5)(conv2)
		pool2 = MaxPooling1D(pool_size=2)(drop2)
		flat2 = Flatten()(pool2)

		# merge
		merged = concatenate([flat1, flat2])
		# interpretation
		dense1 = Dense(100, activation='relu')(merged)
		outputs = Dense(n_outputs, activation='softmax')(dense1)
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		# save a plot of the model
		# plot_model(model, show_shapes=True, to_file='multichannel.png')
		# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(self.learning_rate),
					  metrics=[self.metric])
		self.model = model
		return model

	# def fit(self, trainX, trainy, valX, valy):
	# 	self.model.fit([trainX, trainX], trainy, validation_data=([valX, valX], valy),
	# 				   epochs=self.epochs,
	# 				   batch_size=self.batch_size,
	# 				   verbose=2,
	# 				   callbacks=self.callbacks)
	#
	# def evaluate(self, testX, testy):
	# 	_, accuracy = self.model.evaluate([testX, testX], testy, batch_size=self.batch_size, verbose=2)
	# 	return accuracy

class CNN3Head(BaseModel):
	def build_model(self):
		n_timesteps = self.n_timesteps
		n_features = self.n_channels
		n_outputs =self.n_outputs

		# head 1
		inputs1 = Input(shape=(n_timesteps, n_features))
		conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
		drop1 = Dropout(0.5)(conv1)
		pool1 = MaxPooling1D(pool_size=2)(drop1)
		flat1 = Flatten()(pool1)
		# head 2
		inputs2 = Input(shape=(n_timesteps, n_features))
		conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
		drop2 = Dropout(0.5)(conv2)
		pool2 = MaxPooling1D(pool_size=2)(drop2)
		flat2 = Flatten()(pool2)
		# head 3
		inputs3 = Input(shape=(n_timesteps, n_features))
		conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
		drop3 = Dropout(0.5)(conv3)
		pool3 = MaxPooling1D(pool_size=2)(drop3)
		flat3 = Flatten()(pool3)
		# merge
		merged = concatenate([flat1, flat2, flat3])
		# interpretation
		dense1 = Dense(100, activation='relu')(merged)
		outputs = Dense(n_outputs, activation='softmax')(dense1)
		model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
		# save a plot of the model
		# plot_model(model, show_shapes=True, to_file='multichannel.png')
		# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(self.learning_rate), metrics=['sparse_categorical_accuracy'])
		self.model = model
		return model

	def fit_model(self, trainX, trainy, valX, valy):
		self.model.fit([trainX,trainX,trainX], trainy,
					   # validation_split=0.4,
					  epochs=self.epochs,
					  batch_size=self.batch_size,
					  verbose=2,
					  callbacks=self.callbacks)

	def evaluate(self, testX, testy):
		_, accuracy = self.model.evaluate([testX,testX,testX], testy, batch_size=self.batch_size, verbose=2)
		return accuracy


def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 2, 25, 16
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], 5
 	# head 1
	inputs1 = Input(shape=(n_timesteps,n_features))
	conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# head 2
	inputs2 = Input(shape=(n_timesteps,n_features))
	conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# head 3
	inputs3 = Input(shape=(n_timesteps,n_features))
	conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(100, activation='relu')(merged)
	outputs = Dense(n_outputs, activation='softmax')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# save a plot of the model
	# plot_model(model, show_shapes=True, to_file='multichannel.png')
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit([trainX,trainX,trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	model.save(saved_model_dir/'cnn3heads_origin.h5')
	# evaluate model
	_, accuracy = model.evaluate([testX,testX,testX], testy, batch_size=batch_size, verbose=0)

	return accuracy

if __name__=='__main__':
	# data_dir = Path(r'../data/physionet_sleep/eeg_fpz_cz').resolve()
	data_dir = Path(r'../../data/physionet_sleep/eeg_fpz_cz').resolve()
	train, val, test = prepare_train_test(data_dir=data_dir)
	print(len(train))
	from pysleep.data.data_generator import DataGenerator



	X,y = create_dataset(train[0:3])
	val_X, val_y = create_dataset(val[0:1])
	# train_dl = DataGenerator(train[0:3])
	# evaluate_model(X, y, val_X, val_y)

	if 1:
	# model = CNN1Head(model_name='CNN1Head_train3_test2', epochs=20, learning_rate=0.005, batch_size=32)
		model = CNN3Head(model_name='CNN3Head_train10_test0', epochs=25, learning_rate=0.001, batch_size=16, metric='accuracy')
		model.build_model()
		# hist = model.fit_model(train_dl)
		hist = model.fit_model(X,y, val_X, val_y)# validation_data=(val_X,val_y))

		# hist = model.fit(X,y, val_X, val_y)
		print(hist.history)
		print('done')


	# evaluate_model(X,y, val_X, val_y)