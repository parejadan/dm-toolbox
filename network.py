#!/usr/bin/python

import random
import numpy as np

#learning parameters: weights, biases
#hyper =-parameters: epochs, mini_batch_size, learning rate

class Network(object):
	"sizes = list of each layer's neuron count"
	def __init__(self, sizes):
		self.layer_count = len(sizes) #set network layer count
		self.sizes = sizes #save each layer's specific neuron count
		#biases and weights are randomly initialized using guassian distribution with meu 0 and std 1
		self.biases = [ np.random.randn(y,1) for y in sizes[1:] ] #omits input layer baises
		#weights connects layers, ext. net.weights[1] connects first and seccond layer
		self.weights = [ np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:]) ] #omits input layer's weights

	def feedfarward(self, a):
		"returns network output if 'a' is input"
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"train network using mini-batch stochastic gradient descent \
		--------\
		epochs: # of epochs to train for (steps for gradient descent)\
		mini_batch_size: size of mini batches used for training \
		eta: learning rate \
		test_data: if given, network evaluates network after each epochs of training"
		if test_data: n_test = len(test_data)
		n = len(training_data)

		#random sample partitions of mini_batch_size for each epoch
		for j in  xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[ k:k+mini_batch_size] 
					for k in xrange(0, n, mini_batch_size)
				]
			for batch in mini_batches:
				#how network updates weights and biases with mini_batches
				self.update_mini_batch(batch, eta)
			#useful for tracking progress; significantly slows down training	
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test)
			else:
				print "Epoch{0} complete".format(j)

	def update_mini_batch(self, mini_batch, eta):
		"update network's weights and biases with gradient descent using \
		--------\
		backpropogation on a single mini batch \
		mini_batch: list of tuples (x,y) \
		eta: learning rate"
		#temp matricies for biases and weights that get update
		nabla_b = [ np.zeros(b.shape) for b in  self.biases ]
		nabla_w = [ np.zeros(w.shape) for w in self.weights ]
		#get appropriate gradient for each training example tuple
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = " self.backprop(x, y)" #bottleneck for learning
			nabla_b = [ nb+dnb for nb, dnb in zip( nabla_b, delta_nabla_b) ]
			nabla_w = [ nw+dnw for nw, dnw in zip( nabla_w, delta_nabla_w) ]

		self.biases = [ b-(eta/len(mini_batch))*nb
			for b, nb in zip(self.biases, nabla_b) ]
		self.weight = [ w-(eta/len(mini_batch))*nw
			for w, nw in zip(self.weight, nabla_w) ]

	#compute element wise for vector z
	def sigmoid(z): #compute sigmoid
		return 1.0/(1.0+np.exp(-z))
	def sigmoid_prime(z): #sigmoid function derivative		
		return sigmoid(z)*(1-sigmoid(z))