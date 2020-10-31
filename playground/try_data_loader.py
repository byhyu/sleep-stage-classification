import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

X =np.random.random((1000,300,1))
Y = np.array([np.random.randint(3) for _ in range(1000)])

model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Flatten(input_shape=(300, 1)),
    # tf.keras.layers.Dense(516,input_shape=(300,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# model.fit(train_dataset, epochs=10)
# model.fit(train_examples, train_labels)
model.fit(X, Y)
print('done')