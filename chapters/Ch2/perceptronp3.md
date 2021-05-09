# The Perceptron Algorithm, Part 3

So now we've learned about the basics of the perceptron algorithm, with the limiting assumption that its decision boundary had to pass through the origin. Consequently, we only had to find a single optimal weight vector $w$ and not an additional intercept term $\alpha$. Let's remove that assumption here. 

The way to incorporate the intercept term is clever: we add another __fictitious dimension__ (sometimes called the _bias term_) to every point in our dataset: a 1. In other words, we add a column of ones to our dataset $X$. That way, the corresponding weight found by our optomization algorithm will correspond to our intercept term $\alpha$:

$$
f(x) = w \cdot x + \alpha = \begin{bmatrix} w_1 & w_2 & ... & w_d & \alpha \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ ... \\ x_d \\ 1 \end{bmatrix}
$$

So now our data points are in $\mathbb{R}^{d+1}$-dimensional space, but their final coordinate is 1. This means that all of our data points now lie on a common hyperplane $x_{d+1} = 1$. Now finding a hyperplane in $d+1$-dimensional space (that passes through the origin) will include our intercept (or bias) term!
