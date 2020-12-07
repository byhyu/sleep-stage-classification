from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, activations, layers, losses
from tensorflow.keras.layers import *
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
    model.add(layers.Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    # model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.SpatialDropout1D(rate=0.01))
    model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.SpatialDropout1D(rate=0.01))
    model.add(layers.Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid"))
    model.add(layers.GlobalMaxPool1D())
    # model.add(layers.Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation=activations.softmax))
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'] )
    # model.summary()
    return model

def create_cnn_cnn():
    seq_input = layers.Input(shape=(None, 3000, 1))
    epoch_encoding_model = create_cnn_model1()
    encoded_sequence = layers.TimeDistributed(epoch_encoding_model)
    model = Sequential(layers=[seq_input,
                               encoded_sequence,
                               layers.Convolution1D(64, kernel_size=3, activation='relu',padding='same'),
                               layers.Convolution1D(64, kernel_size=3, activation='relu',padding='same'),
                               layers.Convolution1D(5, kernel_size=3, activation='softmax', padding='same')]
                       )
    model.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def cnn_v0():
    model = Sequential(layers=[
        layers.Convolution1D(16, kernel_size=5, activation='relu', padding='valid', input_shape=(3000, 1)),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.GlobalMaxPool1D(),
        layers.Dropout(rate=0.01),
        layers.Dense(32, activation='relu'),
        layers.Dropout(rate=0.05),
        layers.Dense(5, activation='softmax'), ]
    )
    model.compile(optimizer=optimizers.Adam(0.001), loss="categorical_crossentropy",
                  metrics=['accuracy'])  # ,class_model='categorical'
    return model
def cnn_v01(n_outputs=5):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_v1(lr=0.005):
    model = Sequential(layers=[
        layers.Convolution1D(32, kernel_size=5, strides=1, activation='relu', padding='valid', input_shape=(3000,1)),
        layers.Convolution1D(32, kernel_size=5, strides=1, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(64, kernel_size=25, strides=6, activation='relu', padding='valid', input_shape=(3000, 1)),
        layers.Convolution1D(64, kernel_size=50, strides=6, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.GlobalMaxPool1D(),
        layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.05),
        layers.Dense(5, activation='softmax')]
    )
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy']) #,class_model='categorical'
    return model

def cnn_v2(lr=0.005):
    model = Sequential(layers=[
        layers.Convolution1D(128, kernel_size=50, strides=25, activation='relu', padding='valid', input_shape=(3000,1)),
        # layers.Convolution1D(128, kernel_size=50, strides=25, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=8, strides=8),
        layers.SpatialDropout1D(rate=0.1),
        layers.Convolution1D(128, kernel_size=8, strides=1, activation='relu', padding='valid'),
        layers.Convolution1D(128, kernel_size=8, strides=1, activation='relu', padding='valid'),
        layers.Convolution1D(128, kernel_size=8, strides=1, activation='relu', padding='valid'),
        layers.MaxPool1D(4,4),
        layers.SpatialDropout1D(rate=0.5),
        layers.Flatten(),
        # layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        # layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        # layers.MaxPool1D(pool_size=2),
        # layers.SpatialDropout1D(rate=0.1),
        # layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        # layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        # layers.MaxPool1D(pool_size=2),
        # layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        # layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        # layers.GlobalMaxPool1D(),
        # layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.1),
        # layers.Dense(64, activation='relu'),
        layers.Dropout(rate=0.5),
        layers.Dense(5, activation='softmax')]
    )
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy']) #,class_model='categorical'
    return model

def cnn_base():
    model = Sequential(layers=[
        layers.Convolution1D(16, kernel_size=5, activation='relu', padding='valid', input_shape=(3000,1)),
        layers.Convolution1D(16, kernel_size=5, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(32, kernel_size=3, activation='relu', padding='valid'),
        layers.MaxPool1D(pool_size=2),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.Convolution1D(256, kernel_size=3, activation='relu', padding='valid'),
        layers.GlobalMaxPool1D(),
        layers.Dropout(rate=0.01),
        layers.Dense(64, activation='relu'),
]
    )
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc']) #,class_model='categorical'
    return model

def cnn_cnn():
    seq_input = layers.Input(shape=(10, 3000, 1))
    base_model = cnn_base()
    model = Sequential(layers=[
        seq_input,
        layers.TimeDistributed(base_model),
        layers.Convolution1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.Dropout(rate=0.05),
        layers.Convolution1D(5, kernel_size=3, activation='softmax', padding='same')
    ])
    model.compile(optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #class_model='categorical'
    model.summary()
    return model

def cnn_cnn_1():
    seq_input = layers.Input(shape=(10, 3000, 1))
    base_model = cnn_v1()
    model = Sequential(layers=[
        seq_input,
        layers.TimeDistributed(base_model),
        layers.Convolution1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.SpatialDropout1D(rate=0.01),
        layers.Convolution1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.Dropout(rate=0.05),
        layers.Convolution1D(5, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(5,activation='softmax')
    ])
    model.compile(optimizers.Adam(0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # class_model='categorical'
    model.summary()
    return model

    # for layer in base_mode.layers:
    #     layer.trainable = False
    # encoded_sequence = Sequential(layers=[
    #     layers.TimeDistributed(base_model),
    #     layers.Bidirectional(layers.LSTM)
    # pass

