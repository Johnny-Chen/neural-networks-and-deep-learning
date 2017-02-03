"""tf_cnn.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import cPickle
import gzip
import random

# Third-party libraries
import numpy as np
import tensorflow as tf

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data

def vectorized_result(j):
    e = np.zeros([10,])
    e[j] = 1.0
    return e

def my_shuffle(trd):
    # do some transformation for train data
    #training_inputs = [np.reshape(x, (1, 784)) for x in trd[0]]
    training_inputs = [np.reshape(x, (784,)) for x in trd[0]]
    training_results = [vectorized_result(y) for y in trd[1]]
    trd = zip(training_inputs, training_results)
    random.shuffle(trd)
    return zip(*trd)

    

#### Define the quadratic and cross-entropy cost functions

def quadratic_cost(a, y):
	"""Return the cost associated with an output ``a`` and desired output
	``y``.

	"""
	return 0.5*tf.reduce_sum(tf.square(a-y))

def sigmoid_prime(z):
	"""Deriative of the sigmoid function."""
	return tf.sigmoid(z)*(1-tf.sigmoid(z))

class Network(object):

    def __init__(self, layers):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.layers = layers

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
	for layer in self.layers:
            a = layer.set_inpt(a)
	return a

    def cost(self, a, y):
	""" quadratic cost """
	#####return tf.reduce_mean(tf.reduce_sum(tf.square(a-y), 1))
	#####return -tf.reduce_mean(tf.reduce_sum((y*tf.log(a)+(1-y)*tf.log(1-a)), 1))
	# log-likely(google sample)
	return -tf.reduce_sum(y*tf.log(a))

    def evaluate(self, a, y):
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, 1), tf.argmax(y, 1)), tf.float32))

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
	tdi,tdo = my_shuffle(test_data)

	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])

	x_image = tf.reshape(x, [-1,28,28,1])
	a = self.feedforward(x_image)
	c = self.cost(a,y)

	td_eval = self.evaluate(a,y)
	
	#####opt = tf.train.GradientDescentOptimizer(eta)
	opt = tf.train.AdamOptimizer(1e-4)
	train = opt.minimize(c)

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    #print sess.run(self.layers[3].bias)
	    #print sess.run(self.layers[2].bias)
	    print sess.run(self.layers[1].bias)
	    print sess.run(self.layers[0].bias)

            for j in range(epochs):
		inputs,outputs = my_shuffle(training_data)
		num_training_batches = len(inputs) / mini_batch_size
		for i in range(num_training_batches):
		    sess.run(train, feed_dict = {x:inputs[mini_batch_size*i:mini_batch_size*(i+1)], y:outputs[mini_batch_size*i:mini_batch_size*(i+1)]})

		print sess.run(self.layers[0].bias)
		eval_results = sess.run(td_eval, feed_dict = {x:tdi[0:5000], y:tdo[0:5000]})
		print j , eval_results


class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, pool_shape=(2, 2), activation_fn=None):
	""" the filter_shape should be like [5,5,1,32] """
	self.weight = weight_variable(filter_shape)
	self.bias = bias_variable([filter_shape[3]])

    def set_inpt(self, x_image):
	print "liujiang conv and pool"
	conv = tf.nn.relu(conv2d(x_image, self.weight) + self.bias)
	return max_pool_2x2(conv)


class FullyConnectedLayer(object):

    def __init__(self, filter_shape, pool_shape=(2, 2), activation_fn=None):
	""" the filter_shape should be like [7*7*64,1024] """
	self.shape = filter_shape
	self.weight = weight_variable(filter_shape)
	self.bias = bias_variable([filter_shape[1]])

    def set_inpt(self, x_image):
	print "liujiang fc"
	flat = tf.reshape(x_image, [-1, self.shape[0]])
	return tf.nn.relu(tf.matmul(flat, self.weight) + self.bias)

class SoftmaxLayer(object):

    def __init__(self, filter_shape, pool_shape=(2, 2), activation_fn=None):
	""" the filter_shape should be like [1024,10] """
	self.shape = filter_shape
	self.weight = weight_variable(filter_shape)
	self.bias = bias_variable([filter_shape[1]])

    def set_inpt(self, x_image):
	print "liujiang softmax"
	# drop out firstly
	drop =  tf.nn.dropout(x_image, 0.5)#####
	return tf.nn.softmax(tf.matmul(drop, self.weight) + self.bias)







def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

