from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, activations, layers, losses

def create_baseline_dnn_model():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(3000, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    model.summary()
    return model

def create_dnn_model2():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(3000, 1)))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    model.summary()
    return model

def create_cnn_model1():
    model = Sequential()
    # model.add(layers.Flatten(input_shape=(3000, 1)))
    model.add(layers.Input(shape=(3000,1)))
    model.add(layers.Convolution1D(16,kernel_size=5, activation=activations.relu, padding="valid"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    # model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    model.summary()
    return model
