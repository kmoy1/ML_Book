# MNIST

The MNIST dataset is a very well-known dataset that contains handwritten digits, as well as the number they represent. It is commonly used to evaluate the quality of classifiers. 

Each handwritten digit is an image that is represented by a 28x28 matrix with greyscale levels as its values: each matrix value is an integer from 0 to 255. We want to convert each matrix to a length-784 vector. Thus, each image is represented as a point in 784-dimensional space. This is pretty much the universal standard representation for ANY data point with some number of features: a point in (number of features)-dimensional space.

Unfortunately, it's pretty hard to imagine anything in more than 3 dimensions, much less 784. But it turns out a lot of the concepts that apply in the familiar 2D and 3D spaces carry over to $n$ dimensions. 






