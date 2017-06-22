# main.py
"""
Main file to call methods in mnist_dataset_compress.py and neural_net.py

This code has been developed from "Neural networks and deep learning" by Michael Nielsen
"""

#Libraries

import neural_net
import mnist_dataset_compress
import math
import mpmath
import numpy as np
import scipy
from numpy import zeros, arange, random
from matplotlib.pyplot import plot, show, figure, title


a = 0.5         # learning rate
b =  5          # regularisation parameter 
x=0

num_samples = 396


for x in range(1):          #loop for consecutive testing

    #num_samples = num_samples - 4 
    mean = 0
    sigma = (1/784)        
    A = np.random.normal(mean, sigma, (num_samples,784))

    #Load data 
    new_training_data, new_validation_data = mnist_dataset_compress.load_data_wrapper(num_samples , x, A)
     
    #Build neural network    
    nueral_net = neural_net.neural_network([num_samples,25,10], cost_function = neural_net.cross_entropy_cost) 
    
    nueral_net.StochasticGradientDescent(new_training_data, new_validation_data, 30, 10, a,lambda_ = b,classification_data=new_validation_data)
        
     
   
    

