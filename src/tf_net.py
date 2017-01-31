"""network2.py
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

    def __init__(self, sizes):
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
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [tf.Variable(tf.random_normal([y], dtype=tf.float32)) for y in sizes[1:]]
        self.weights = [tf.Variable(tf.random_normal([x,y], dtype=tf.float32))
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #self.biases = [tf.Variable(tf.zeros([y], tf.float32)) for y in sizes[1:]]
        #self.weights = [tf.Variable(tf.zeros([x,y], tf.float32))
        #               for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
	for w,b in zip(self.weights, self.biases):
            a = tf.sigmoid(tf.matmul(a, w) + b)
	return a

    def cost(self, a, y):
	""" quadratic cost """
	return tf.reduce_mean(tf.reduce_sum(tf.square(a-y), 1))

    def evaluate(self, a, y):
	return sum(int(np.argmax(i) == np.argmax(j)) for i,j in zip(a,y))

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

	a = self.feedforward(x)
	c = self.cost(a,y)
	
	opt = tf.train.GradientDescentOptimizer(eta)
	train = opt.minimize(c)

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())

            for j in range(epochs):
		inputs,outputs = my_shuffle(training_data)
		for i in range(5000):
		    sess.run(train, feed_dict = {x:inputs[10*i:10*(i+1)], y:outputs[10*i:10*(i+1)]})
		td_results = sess.run(a, feed_dict = {x:tdi})
		print j , self.evaluate(td_results, tdo)
