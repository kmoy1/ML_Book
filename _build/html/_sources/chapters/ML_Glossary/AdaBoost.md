# AdaBoost

__AdaBoost__ is a popular ensemble learning method for classifiers. It is short for "adaptive boosting". The Big Idea: AdaBoost iteratively learns rom the mistakes of weak classifiers and turns them into strong ones.

AdaBoost is a special kind of [boosting](./Boosting.md) algorithm.

A single classifier may not be able to accurately predict the class of an object, but when we group multiple weak classifiers with each one progressively learning from the others' wrongly classified objects, we can build one such strong model. 

