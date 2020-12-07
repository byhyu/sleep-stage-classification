import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from .base_model import BaseModel

CONFIG = {'sample_rate':100,
          'seq_length':10}

class CNN_LSTM(BaseModel):
    def __init__(self,
                 config=CONFIG,
                 output_dir='./output'):
        self.config= config
        self.output_dir = output_dir
        # super().__init__()

    def convert_input_to_sequence_input(self, inputs):
        """ Reshape the input from (batch_size * seq_length, input_dim) to
        (batch_size, seq_length, input_dim)
        :param inputs:
        :return:
        """
        input_dim = inputs.shape[-1].value
        seq_inputs = tf.reshape(inputs, shape=[-1,self.config['seq_length'], input_dim])

    def create_rnn(self):
        def _create_rnn_cell(n_units):
            cell = LSTMCell(n_units=n_units,
                            )

    def build_cnn(self):
        first_filter_size = int(self.config['sample_rate'] / 2.0)
        first_filter_stride = int(self.config['sample_rate'] / 16.0)
        model = Sequential(layers=[
            Input(shape=[3000,1]),
            Conv1D(128, first_filter_size, first_filter_stride),
            BatchNormalization(),
            ReLU(),
            MaxPooling1D(8, 8),
            Dropout(0.5),
            Conv1D(128, 8, 1),
            BatchNormalization(),
            ReLU(),
            Conv1D(128, 8, 1),
            BatchNormalization(),
            ReLU(),
            Conv1D(128, 8, 1),
            BatchNormalization(),
            ReLU(),
            MaxPooling1D(4, 4),
            Flatten(),
            Dropout(0.5),])
        return model

    def build_model(self):
        cnn = self.build_cnn()
        model = Sequential(layers=[
            TimeDistributed(cnn),
            LSTM(128),
            Dropout(0.5),
            Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
