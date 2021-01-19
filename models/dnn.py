from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, activations, layers, losses
from tensorflow.keras.layers import *
from .base_model import BaseModel

def DNN(BaseModel):
    def build_model(self):
        n_timesteps = self.n_timesteps
        n_features = self.n_channels
        n_outputs = self.n_outputs
        model = Sequential()
        model.add(layers.Flatten(input_shape=(n_timesteps, n_features)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_outputs, activation=activations.softmax))
        model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model

# def create_dnn_model():
#     model = Sequential()
#     model.add(layers.Flatten(input_shape=(3000, 1)))
#     model.add(Dense(128,activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(5, activation=activations.softmax))
#     model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
#     model.summary()
#     return model
