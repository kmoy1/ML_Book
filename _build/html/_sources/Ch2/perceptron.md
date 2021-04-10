# The Perceptron Algorithm

The perceptron algorithm is an obsolete but instructive algorithm. It is slow, but will guarantee finding a separating boundary for a linearly separable dataset. The bad part is, it is a little _too_ reliant on the dataset being linearly separable: it doesn't even terminate if it isn't. 

The basic concept behind the perceptron algorithm is still used by many popular classifiers today, from logistic regression to neural networks. It is one of the first machine learning algorithms to utilize __numerical optomization__: specifically, __gradient descent__, one of the core algorithms in ML. 

Consider $n$ sample points denoted as $X_1, X_2, ... , X_n$ (each is a row in the design matrix $X$). We also have corresponding labels $y_1, y_2, ..., y_n$, where $y_i = 1$ if $X_i$ is in class C, and $y_i = -1$ if $X_i$ is not in class C. 

To start off, we'll make a big simplifying assumption: we assume the decision boundary passes through the origin. In other words, we assume $\alpha = 0$. Yes, this is a big limitation on the boundary but we'll fix it very quickly with a simple adjustment later. 

Now, our goal is to find the optimal normal vector $w$. Specifically, we want $w$ such that every point $X_i$ is classified correctly. In mathematical terms, we want the __signed distance__ $X_i \cdot w \ge 0$ if $y_i = 1$, and $X_i \cdot w \le 0$ if $y_i = -1$. So signed distances for points in class C are positive, and negative for those not in class C, just as we expect. 

We can actually further simplify the previous two assumptions into this single _constraint_
$$
y_iX_i \cdot w \ge 0
$$

for all $i$.

## Risk Function

Now we define a __risk function__ $R$ which takes in a classifier $r*$. The idea is that the risk function accumulates positive risk for each constraint that is violated when the classifier classifies. So basically the risk function is a function that "evaluates" a classifier with a score: the lower the score, the better. Accordingly, we will use optomization to choose the optimal $w$ that minimizes $R$. 

```{note}
The risk function is also called the _objective function_ since we want to minimize it for our classifier.  
```

Part of the risk function involves individually scoring whether each data point is classified correctly or not. For this we use a __loss function__ $L(z, y_i) defined as (for the two-class case)

$$
L(z, y_i) = 
    \begin{cases} 
      0 & y_iz \ge 0 \\
      -y_iz & y_iz < 0 \\  
   \end{cases}
$$

where, for data point $i$, $z$ is our prediction, and $y_i$ is our truth label. Notice that if $y_iz < 0$, then we have an incorrect classification, because a correct classification would have $y_i$ and $z$ having the same sign. However, on a correct prediction, our loss is 0. So now __risk is just the average of loss over all points.__ Formally:

$$
R(w) = \frac{1}{n}\sum_{i=1}^{n}L(X_i\cdot w_i, y_i)
$$

But since $L(X_i \cdot w_i, y_i) = 0$ for a correct prediction, we're really just taking the summing over all misclassified points, which allows us to explicitly write out the loss function:

$$
R(w) = \frac{1}{n}\sum_{i \in V}-y_iX_i \cdot w
$$

where $V$ is the set of indices that were misclassified: i.e. $y_iX_i \cdot w < 0$. 

So a perfect classifier $w$ would have $R(w) = 0$. Of course, this is not always possible, so we want to try to find $w$ such that $R(w)$ is minimized. We can formally denote this problem as an optomization problem:

$$
\underset{w}{\arg\min} R(w)
$$

We can think of this optomization as finding the point $w$ that minimizes $R(w)$ in feature space. Note that in this case, the origin $w = 0$ technically minimizes the risk function but is obviously useless. So we really want the optimal _nonzero_ $w$.

Unfortunately, because there's so much variation in what the risk function could look like that depends on the initial dataset, there's not really a closed-form solution here- we don't have an explicit works-all-the-time formula for an optimal $w$. However, that does not mean we can't find one. The means of finding it is one of the most important algorithms ever created for machine learning, called gradient descent. We'll cover this in detail in the next chapter. 