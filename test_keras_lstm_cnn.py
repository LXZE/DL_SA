import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Bidirectional, CuDNNGRU
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import backend as K

import numpy as np

learning_rate = 0.0001
num_steps = 10
batch_size = 128
epochs = 10
# display_step = 100

data_dim = 300
timesteps = 10
num_classes = 2

x_train = np.random.random((60000, timesteps, data_dim))
y_train = np.random.random((60000, num_classes))
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

dropout = 0.5

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
print('train_x shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

print(x_train[0])

# model = Sequential()
# model.add(Bidirectional(CuDNNGRU(32, return_sequences=True, input_shape=(timesteps, data_dim))))
# model.add(Bidirectional(CuDNNGRU(32)))
# model.add(Dense(2, activation='softmax'))

# adam = optimizers.Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
# print(model.summary())


# train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
# test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_classes)

# model = Sequential()
# model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28, 1), bias_initializer='RandomNormal'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5,5), activation='relu', padding='same', bias_initializer='RandomNormal'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(dropout))
# model.add(Dense(128, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# adam = optimizers.Adam(lr=learning_rate)

# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model_name = 'model.h5'
# if os.path.exists(model_name):
# 	model.load_weights(model_name)
# 	print("Loaded model from disk")
# else:
# 	model.fit(train_x, train_y, batch_size=batch_size, epochs=num_steps)
# 	model.save_weights("model.h5")
# 	print("Saved model to disk")
# score = model.evaluate(test_x, test_y, batch_size=batch_size)
# print("{}: {}".format(model.metrics_names[0], score[0]))
# print("{}: {}%".format(model.metrics_names[1], score[1]*100))