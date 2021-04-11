# The MNIST dataset

The MNIST dataset is a well-known dataset that contains handwritten digits, as well as the number they represent. It is commonly used to evaluate the quality of classifiers. 

Each handwritten digit is an image that is represented by a 28x28 matrix with greyscale levels as its values: each matrix value is an integer from 0 to 255.

So a handwritten 5, for example, will be represented as

from scipy import io
import csv
import os

def load_data(name):
    """Return matrix NAME as a dictionary.
    Possible parameter options: cifar10_data, spam_data, mnist_data
    """
    data = io.loadmat(name + ".mat")
    return data

MNIST_dataset = load_data('mnist_data')
drawn_five = MNIST_dataset['training_data'][0].reshape((28,28))
mx = 3

for line in drawn_five:
    print(" ".join(["{:<{mx}}".format(elmt, mx=mx) for elmt in line]))


Pretty cool, huh? 

We want to convert each matrix to a length-784 vector. Thus, each image is represented as a point in 784-dimensional space. This is pretty much the universal standard representation for ANY data point with some number of features: a point in (number of features)-dimensional space. Of course, more sophisticated representations exist, but this will be more than enough for now. 

Unfortunately, it's pretty hard to imagine anything in more than 3 dimensions, much less 784. But it turns out a lot of the concepts that apply in the familiar 2D and 3D spaces carry over to $n$ dimensions. Our decision boundary in 784-dimensional space is still linear, but rather than a line, it is a __hyperplane__. __Given our _ambient space (the space for our points, basically) is $d$ dimensions, a hyperplane in that space will always be $d-1$ dimensions.__ For example, 2D space will have a decision boundary in 1 dimension: a line. 1D space will have a DB in 0 dimensions: a single point. 3D space, 2D plane. In $n$-dimensional space, the concept still stays the same: the ambient space is being cut into two pieces by an $n-1$-dimensional hyperplane, where one side has predict IN CLASS, other side predict NOT IN CLASS. 