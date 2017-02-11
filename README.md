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

I tested this method by decreasing the number of measurements taken from the original image to see how the number of samples taken from the original image effected how the neural network was able to recognize the images. I started the test  by using all the samples in the image (784) and continued the test until I only took  4 samples decrementing by 4 samples each time. I then plotted the corresponding recognition accuracy against the number of samples used:


<img width="639" alt="screen shot 2017-02-06 at 16 17 27" src="https://cloud.githubusercontent.com/assets/18538034/22853826/c17aab6a-f057-11e6-949f-4c02d62e85f7.png">


Analysis:

As we can see from the graph above the recognition accuracy remains constant at approximately 90% accuracy until we only take 100 samples (12.75% of samples from original image). This thus represents an ability to reduce the number of samples taken by  87.2% and still achieve the same recognition accuracy as If all samples were taken.

If we only take 40 samples out of the full image (784 samples) which represents a 95% reduction of samples taken from the image, we can still get a recognition accuracy of 80.79% percent.

