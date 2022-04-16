# Boosting

The __Boosting__ algorithm is a special type of ensemble learning algorithm that tries to build a strong model by correcting the mistakes of several weaker ones.

## The Boosting Algorithm

First, create an initial model, and try running it on training data. Error should be trash. On the next iteration, we create a second improved model by reducing errors from the previous one. We do this sequentially until the training data is predicted perfectly or we reach some arbitrary threshold. 

A very popular type of boosting algorithm is [AdaBoost](./AdaBoost.md).