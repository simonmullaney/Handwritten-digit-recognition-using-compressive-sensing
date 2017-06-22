# neural_net.py
"""
File for neural network to recognise the compressed images in the MNIST dataset which have been loaded
and compressed in mnist_dataset_compress.py. These images have been compressed using a 2-D DCT bias function,
they have then been sensed using a random Gaussian sensing matrix. This matrix selects num_samples from the image 
after the 2D DCT basis function has been applied to it. The key features of the MNIST images should then be stored in
the smaller vector num_samples. This smaller vector is then used to train the neural recognition to see if the
neural network can correctly classify the compressed dataset.


"""

# Libraries

import json
import random
import sys
import png
import bigfloat
import mpmath
import numpy as np  	#library for doing linear algebra
import pylab as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso
from numpy import zeros, arange, random
from decimal import *
from mpmath import *

 
""" Definition of sigmoid output function """
 
def sigmoid_fn(x):						
    sigmoid = 1.0/(1.0+np.exp(-x))						#sigmoid function
    
    return sigmoid
    
""" derivative of the sigmoid output function """    
def sigmoid_prime_fn(x):				
    sigmoid_prime = sigmoid_fn(x)*(1-sigmoid_fn(x))     #Derivative of the sigmoid function
    
    return sigmoid_prime
    
    
"""Define quadratic cost function, this was the first cost function used on the first verision of the neural network """   
class quadratic_cost(object):

    @staticmethod
    def delta_quadratic(z, a, y):
         # calculates the error delta from the output layer.
        delta_quadratic = (a-y)*sigmoid_prime_fn(z)
        return delta_quadratic

    @staticmethod
    def fn_quadratic(a, y):
        #calculates the cost for an output `a`, for a desired output `y` 
        cost_quadratic  = 0.5*np.linalg.norm(a-y)**2
        return cost_quadratic

""" 
	Define cross-entropy cost function, this was the second and final cost function used. It prevents the 
    learning slowdown that the quadratic cost function experiences. This is because the way weights and biases learn 
    is dependant on the error in the network. Thus the larger the error the faster the learning.
    
"""
class cross_entropy_cost(object):

    @staticmethod
    def delta_cross_entropy(a, y):
        # calculates the error delta from the output layer.
        delta_cross = (a-y)

        return delta_cross

    @staticmethod
    def fn_cross_entropy(a, y):
    
        #calculates the cost for an output `a`, for a desired output `y`
        cross_cost = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        
        return cross_cost

# Neural Network class

class neural_network(object):


    def __init__(self, network_topology, cost_function = cross_entropy_cost):
        """network_topology details the number of layers and nodes in the neural network.  The biases and weights for the network
        are initialized randomly
        """
        self.network_topology = network_topology
        self.num_network_layers = len(network_topology)
        self.cost_function = cost_function
        self.weight_init()
        
    def feedforward(self, activation):
        """Return the output of the network if a is input."""
        
        for biases, weights in zip(self.biases, self.weights):
        
            w_x = np.dot(weights, activation)
            activation = sigmoid_fn(w_x + biases)
        return activation
            
        
	#Stochastic Gradient Descent method
    def StochasticGradientDescent(self, new_training_data, new_validation_data, num_epochs, batch_size, eta, lambda_ = 0.0, classification_data=None):
            
        len_classification_data = len(classification_data)
        len_new_training_data = len(new_training_data)
        evaluation_accuracy = []
                             
        for j in range(num_epochs):
            random.shuffle(new_training_data)
            
            training_inputs_batches = [new_training_data[k : k+batch_size] for k in range(0, len_new_training_data, batch_size)]  
            
            for batch in training_inputs_batches:
            
                self.update_mini_batch(batch, eta, lambda_, len(new_training_data))			#update minibatches
                
                
            print("Epoch training number %s complete" % j)
            
            
            classification_accuracy = self.accuracy(classification_data)
            evaluation_accuracy.append(classification_accuracy)
            print("Classification accuracy on the evaluation dataset: {} / {}".format(self.accuracy(classification_data), len_classification_data))
                
            #Write accuracy to RESULTS_MACHINE_LEARNING.rtf  
            f = open('/Users/simonmullaney/Desktop/RESULTS_MACHINE_LEARNING.rtf', 'a')
            f.write("\n")
            f.write(str(self.accuracy(classification_data)))
            f.write("\n")
                    
            
            print
        return  evaluation_accuracy, num_epochs
        
    def backpropagation(self, x, y):
        """
        Backpropagation algorithm calculates the gradient of the cost function
        nabla_b and nabla_w are layer by layer arrays
        """
        
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activ = x
        activ_array = [x] 		# list to store all the activations, layer by layer
        z_array = [] 			# list to store all the z vectors, layer by layer
    
        for bias, weights in zip(self.biases, self.weights):
        
            z = np.dot(weights, activ) + bias       # (Weights)*(inputs) + bias
            z_array.append(z)						# array of z values
            activ = sigmoid_fn(z)					# activation = sigma(z), where sigma is the sigmoid output function
            activ_array.append(activ)				# array of activations
        
        # backward pass
        delta_cross_entropy = (self.cost_function).delta_cross_entropy(activ_array[-1], y)
        nabla_bias[-1] = delta_cross_entropy
        nabla_weights[-1] = np.dot(delta_cross_entropy, activ_array[-2].transpose())
        
        
        # l = -1 means the last layer of neurons
        
        for l in range(2, self.num_network_layers):
            z = z_array[-l]
            sigmoid_prime = sigmoid_prime_fn(z)
            delta_cross_entropy = np.dot(self.weights[-l+1].transpose(), delta_cross_entropy) * sigmoid_prime
            nabla_bias[-l] = delta_cross_entropy
            nabla_weights[-l] = np.dot(delta_cross_entropy, activ_array[-l-1].transpose())
            
            
        return (nabla_bias, nabla_weights) 
            
    def weight_init(self):
        """ Weights are initialised using a Gaussian random distribution with a mean of 0 and 
        a standard deviation 1 over the square root of the number of weights connecting to the same neuron.
        The biases are also initialised using a Gaussian random distribution with a mean of 0 and 
        a standard deviation 1.
        No biases are set for the input layer.
        """
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.network_topology[:-1], self.network_topology[1:])]
        
        self.biases = [np.random.randn(y, 1) for y in self.network_topology[1:]]       
          

    def update_mini_batch(self, batch, eta, lambda_, n):
        """"
        Method to update the neural networks weights and biases. This is done by applying the gradient descent using 
        backpropagation on a single minibatch
        """
        
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        
        
        #print(mini_batch)
        for x, y in batch:
            delta_nabla_bias, delta_nabla_weights = self.backpropagation(x, y)
            
            nabla_bias = [nb+dnb for nb, dnb in zip(nabla_bias, delta_nabla_bias)]
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
        
        self.weights = [(1-eta*(lambda_/n))*w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_weights)] 	#eta is the learning rate
        self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_bias)]						#lambda = regularisation parameter

    
    

    def accuracy(self, data):
        """
        Calculating the accuracy of the neural network,returns the number of times the nerual network
        outputs a correct result. The output of the neural network is the neuron which has 
        the highest activation in the final layer
        """
        
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data] 
        x = sum(int(x == y) for (x, y) in results)          
                    
        return x
        


    
