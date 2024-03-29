import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import backend as K

mnist = tf.keras.datasets.mnist


from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"]
import numpy as np

learning_rate = 0.0001
num_steps = 10
batch_size = 128
# display_step = 100

num_input = 784 # MNIST data input (img shape: 28*28)
img_row, img_col = 28, 28
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.50 # Dropout, probability to keep units

(train_x, train_y), (test_x, test_y) = mnist.load_data()

if K.image_data_format() == 'channels_first':
	train_x = train_x.reshape(train_x.shape[0], 1, img_row, img_col)
	test_x = test_x.reshape(test_x.shape[0], 1, img_row, img_col)
	input_shape = (1, img_row, img_col)
else:
	train_x = train_x.reshape(train_x.shape[0], img_row, img_col, 1)
	test_x = test_x.reshape(test_x.shape[0], img_row, img_col, 1)
	input_shape = (img_row, img_col, 1)

train_x = train_x.astype('float16')
test_x = test_x.astype('float16')
print('train_x shape:', train_x.shape)
print(train_x.shape[0], 'train samples')
print(test_x.shape[0], 'test samples')


train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_classes)

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28, 1), bias_initializer='RandomNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5,5), activation='relu', padding='same', bias_initializer='RandomNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

adam = optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_name = 'model.h5'
if os.path.exists(model_name):
	model.load_weights(model_name)
	print("Loaded model from disk")
else:
	model.fit(train_x, train_y, batch_size=batch_size, epochs=num_steps)
	model.save_weights("model.h5")
	print("Saved model to disk")
score = model.evaluate(test_x, test_y, batch_size=batch_size)
print("{}: {}".format(model.metrics_names[0], score[0]))
print("{}: {}%".format(model.metrics_names[1], score[1]*100))

def gen_image(arr):
	two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
	plt.imshow(two_d, interpolation='nearest')
	return plt

rand = np.random.choice(len(test_x), 10, replace=False)
for i in rand:
	pre_x, pre_y = test_x[i], test_y[i]
	pre_x = pre_x.reshape(1, img_row, img_col, 1)
	res_y = model.predict_classes(pre_x, batch_size=1)
	# gen_image(pre_x).show()
	print('Prediction: {}'.format(res_y[0]))
	print('Result: {}'.format(str(np.where(pre_y==1.)[0])[1]))



# from __future__ import division, print_function, absolute_import
# import tensorflow as tf
# from tensorflow.contrib import rnn

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"]
# import numpy as np

# learning_rate = 0.001
# num_steps = 1000
# batch_size = 128
# display_step = 100

# num_input = 784 # MNIST data input (img shape: 28*28)
# num_classes = 10 # MNIST total classes (0-9 digits)
# dropout = 0.50 # Dropout, probability to keep units

# X = tf.placeholder(tf.float32, [None, num_input])
# Y = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# def conv2d(x, W, b, strides=1):
# 	# Conv2D wrapper, with bias and relu activation
# 	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
# 	x = tf.nn.bias_add(x, b)
# 	return tf.nn.relu(x)

# def maxpool2d(x, k=2):
# 	# MaxPool2D wrapper
# 	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
# 						  padding='SAME')

# def conv_net(x, weights, biases, dropout):
# 	# MNIST data input is a 1-D vector of 784 features (28*28 pixels)
# 	# Reshape to match picture format [Height x Width x Channel]
# 	# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
# 	x = tf.reshape(x, shape=[-1, 28, 28, 1])

# 	# Convolution Layer
# 	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
# 	# Max Pooling (down-sampling)
# 	conv1 = maxpool2d(conv1, k=2)

# 	# Convolution Layer
# 	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
# 	# Max Pooling (down-sampling)
# 	conv2 = maxpool2d(conv2, k=2)

# 	# Fully connected layer
# 	# Reshape conv2 output to fit fully connected layer input
# 	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
# 	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
# 	fc1 = tf.nn.relu(fc1)
# 	# Apply Dropout
# 	fc1 = tf.nn.dropout(fc1, dropout)

# 	# Output, class prediction
# 	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
# 	return out

# weights = {
# 	# 5x5 conv, 1 input, 32 outputs
# 	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
# 	# 5x5 conv, 32 inputs, 64 outputs
# 	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
# 	# fully connected, 7*7*64 inputs, 1024 outputs
# 	'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
# 	# 1024 inputs, 10 outputs (class prediction)
# 	'out': tf.Variable(tf.random_normal([1024, num_classes]))
# }

# biases = {
# 	'bc1': tf.Variable(tf.random_normal([32])),
# 	'bc2': tf.Variable(tf.random_normal([64])),
# 	'bd1': tf.Variable(tf.random_normal([1024])),
# 	'out': tf.Variable(tf.random_normal([num_classes]))
# }

# logits = conv_net(X, weights, biases, keep_prob)
# prediction = tf.nn.softmax(logits)

# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# 	logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)


# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

# # Start training
# # with tf.Session() as sess:
# sess = tf.Session()

# # Run the initializer
# sess.run(init)

# for step in range(1, num_steps+1):
# 	batch_x, batch_y = mnist.train.next_batch(batch_size)
# 	# Run optimization op (backprop)
# 	sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
# 	if step % display_step == 0 or step == 1:
# 		# Calculate batch loss and accuracy
# 		loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
# 															 Y: batch_y,
# 															 keep_prob: 1.0})
# 		print("Step " + str(step) + ", Minibatch Loss= " + \
# 			  "{:.4f}".format(loss) + ", Training Accuracy= " + \
# 			  "{:.3f}".format(acc))

# print("Optimization Finished!")

# # Calculate accuracy for 256 MNIST test images
# print("Testing Accuracy:", \
# 	sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
# 								  Y: mnist.test.labels[:256],
# 								  keep_prob: 1.0}))
# file_writer = tf.summary.FileWriter('logs', sess.graph)
# file_writer.close()

# def gen_image(arr):
# 	two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
# 	plt.imshow(two_d, interpolation='nearest')
# 	return plt

# for i in range(0, 10):
# 	pre_x, pre_y = mnist.train.next_batch(1)
# 	res_y = sess.run(prediction, feed_dict={X: pre_x, keep_prob:1.0})
# 	gen_image(pre_x).show()
# 	print('Prediction: {}'.format(str(np.where(res_y[0]==1)[0])[1]))
# 	print('Result: {}'.format(str(np.where(pre_y[0]==1)[0])[1]))