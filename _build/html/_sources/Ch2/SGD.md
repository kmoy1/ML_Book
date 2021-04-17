# Stochastic Gradient Descent

One big problem with the perceptron algorithm, though, is that it's very slow! Each step takes $O(nd)$ time, where $n$ is the number of sample points and $d$ is the number of features. How can we fix this? 

Remember that in normal gradient descent, we need to calculate the gradient as $\sum_{i \in V}y_iX_i$. This requires us to sum through all misclassified points at each iteration. Stochastic gradient descent improves on this time sink drastically by only requiring us to pick ONE misclassified point $X_i$ at each iteration. Then, instead of doing gradient descent on $R$, all we have to do is gradient descent on the loss function $L(X_i \cdot w, y_i)$. 

Now, each iteration of gradient descent takes around $O(d)$ time instead of $O(nd)$: a MASSIVE improvement, especially with a very large dataset! However, do note there's always a tradeoff: we're not using complete information of all misclassified points, so we rely more on chance to get the optimal $w$ we want. 

<!-- TODO: Include pseudocode for SGD -->