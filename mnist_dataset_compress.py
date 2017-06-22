# mnist_dataset_compress.py
"""
A file used to load the MNIST dataset, this loads three arrays: training_data, validation_data and test_data. 

These three arrays then undergo a compressive sensing operation that employs a 2-D DCT bias function to transform
the images to the frequency domain. A random Gaussian matrix is used as the sensing matrix to select values from
the images which were transformed to the frequency domain. This compressed version of the vector is then passed to
our neural network for recognition


"""

# Libraries

import pickle as cPickle
import gzip

import numpy as np
import pylab as plt
from matplotlib.pyplot import plot, show, figure, title
import matplotlib as plt1
import math
import mpmath
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import scipy.optimize as spopt
import scipy.ndimage as spimg
import scipy.fftpack as spfft

from PIL import Image
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso
from numpy import zeros, arange, random
from sklearn import linear_model
from mpmath import *
from bigfloat import BigFloat, precision
from decimal import *
from numpy import linalg as la
from math import cos,sqrt,pi


def load_data(num_samples, x, A):
    
    """
    This method loads the MNIST dataset into three arrays; training_data, validation_data, and test_data.
    The arrays then undergo compressive sampling to store the key features of the MNIST image in a smaller set of samples
    """
    
    #Loading the MNIST dataset
    f = gzip.open('/Users/simonmullaney/Desktop/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    
    no_training_images = 50000
    i = 0
    
    #Code to compressive sense training_data to new_training_data sampled at num_samples
    
    new_training_data = np.zeros((2,no_training_images), dtype = object) 	# Initialising new_training_data array
    x = np.zeros((num_samples,1))
       
    k = 0
    for k in range(no_training_images):					#making first row of new_training_data = empty column vector for image
        new_training_data[0][k] = x
        
    k = 0
    for k in range(no_training_images):						#associating second row of new training values to correct validation labels
        new_training_data[1][k] = training_data[1][k]
        
    
    i = 0
    for i in range(no_training_images):						# Compressing Training data
      
        Z = np.array(training_data[0][i])
        Z = Z.reshape((784,1))
        Z = dct(dct(Z.T).T) 

        #code to visualise image after DCT
        """
        Z = Z.reshape((28,28))
        im = plt.imshow(Z,cmap= 'gray')
        plt.colorbar(im, orientation='horizontal')
        plt.show()
        """
    
        #Y =A∗X 
        new_training_data[0][i] = np.dot(A,Z)

   
    
    #Code to compressive sense validation_data to new_validation_data sampled at num_samples 
    
    no_validation_images = 10000
        
    new_validation_data = np.zeros((2,no_validation_images), dtype = object)       # Initialising new_validation_data array
    x = np.zeros((num_samples,1))
    
    k =0
    for k in range(no_validation_images):				#making first row of new_validation_data = empty column vector for image
        new_validation_data[0][k] = x
        
    k = 0
    for k in range(no_validation_images):					#associating second row of new training values to correct validation labels
        new_validation_data[1][k] = validation_data[1][k]
    
    i = 0
    for i in range(no_validation_images):

        Z = np.array(validation_data[0][i])
        Z = Z.reshape((784,1))
        Z = dct(dct(Z.T).T) 
        
        #Y =A∗X
        x = np.dot(A,Z)   
        new_validation_data[0][i] = np.dot(A,Z) 
            
         
    f.close()
    new_training_data = tuple(new_training_data)
    
    
    return (A,new_training_data,new_validation_data)
    

def vector_result(a):
    """Puts a one in the ath position and zeroes elsewhere"""
    x = np.zeros((10, 1))
    x[a] = 1.0
    return x

def load_data_wrapper(num_samples, x, A):
    """Return a tuple containing (new_training_data, new_validation_data)."""
   
       
    A, new_training_data, new_validation_data  = load_data(num_samples , x, A)
    
        
    compressed_train_inputs = [np.reshape(x, (num_samples, 1)) for x in new_training_data[0]]
    compressed_train_results = [vector_result(y) for y in new_training_data[1]]
    new_training_data = list(zip(compressed_train_inputs, compressed_train_results))
    
    
    compressed_val_inputs = [np.reshape(x, (num_samples, 1)) for x in new_validation_data[0]]
    new_validation_data = list(zip(compressed_val_inputs, new_validation_data[1]))
    

    return (new_training_data, new_validation_data)
    


    
 
