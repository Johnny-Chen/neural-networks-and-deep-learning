import tf_net

trd, rd, td = tf_net.load_data_shared()

net = tf_net.Network([784,30,10])
#net = tf_net.Network([784,10])

net.SGD(training_data=trd,eta=0.5, epochs=400, mini_batch_size=50, test_data=td)
#net.SGD(training_data=trd,eta=0.01, epochs=1000, mini_batch_size=50, test_data=td)
