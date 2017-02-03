import cnn
from cnn import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

trd, rd, td = cnn.load_data_shared()

net = cnn.Network([ConvPoolLayer([5,5,1,32]),
			ConvPoolLayer([5,5,32,64]),
			FullyConnectedLayer([7*7*64,100]),
			SoftmaxLayer([100,10])])
#net = tf_net.Network([784,10])

net.SGD(training_data=trd,eta=0.1, epochs=60, mini_batch_size=10, test_data=td)
#net.SGD(training_data=trd,eta=0.01, epochs=1000, mini_batch_size=50, test_data=td)
