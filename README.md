# Handwritten-digit-recognition-using-mahine-learning-&-compressive-sensing


This is a handwritten digit recognition tool. I use a neural network to infer rules on the digits and thus be able to recognise the handwritten digits correctly. I train and test my neural network by using the MNIST dataset (http://yann.lecun.com/exdb/mnist/). I train my neural network with 50,000 images and then test it on a different 10,000 images. Thus results reflect the actual recognition ability of the tool rather than just learning the peculiarities of the dataset it is being trained with.


Compression is achieved by carrying out the operation below:

                      Y = A*Bpq*Ic

Where,
A is the sensing matrix, zero mean Gaussian distribution with Standard deviation = 1/num_samples (in original image),
Y is the compressed version of the data,
Bpq is the 2d DCT basis function,
Ic is the image reordered as a column vector.

The 2d DCT basis basis function Bpq is given by:


<img width="405" alt="screen shot 2017-02-07 at 14 03 20" src="https://cloud.githubusercontent.com/assets/18538034/22853823/b88cdcf8-f057-11e6-9dde-8c029c53f3b6.png">



Results:
