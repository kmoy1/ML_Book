# Supervised vs. Unsupervised Learning

The techniques we will learn throughout this book can be organized into two classes: supervised learning and unsupervised learning.

## Supervised Learning

__Supervised learning__ is training a model where you _know the labels for each data point_: for example, when training the credit card data, we know whether each person actually did default or not. 

Supervised learning can actually be further broken down into two categories: classification and regression. 

In classification, we attempt to assign a datapoint to a class: is it in that class or is it not? Again, our credit card decision boundary performs classification. Another very common example is classifying whether an email is spam or not.

In regression, we attempt to predict a numerical value instead of a discrete number. For example, given the temperature, how many people will go to the beach on that day? Another less-known example of regression is _probability prediction_: given certain features about some person's health, what is the _probability_ that he/she has cancer?

Note that classification and regression are strongly connected: simply assigning a _class threshold_ to the numerical predictions of regression makes it a classification problem as well. For example, if the probability output by our cancer model on a patient is 0.5 or higher, we can simply establish a rule to classify that patient as cancer-positive.

## Unsupervised Learning

On the other hand, __unsupervised learning__ is training a model where you do NOT know the labels for each data point. So you don't necessarily know what you're trying to predict. However, there are still a few ways our data could be made useful, with a few ML techniques!

The first thing we can do is __clustering__: we get a measure of similarity between different data points and attempt to group similar points together to the same class (cluster). For example, given a dataset of thousands of different DNA sequences, which sequences are most similar? Perhaps one cluster inherently belongs to invertebrates and one cluster belongs to vertebrates. Only running the algorithm will tell us: it will be much too difficult and tedious to tell with the human eye.

Another thing we can do is __dimensionality reduction__: perhaps instead of thousands of features, our data points can be completely differentiated using just five features. This is useful not only for runtime, but space and better visualization of similarity/dissimilarity as well. Dimensionality reduction is especially useful when there's a lot of _common features_ that can be simplified down to one or two features that basically "say" the same thing.
