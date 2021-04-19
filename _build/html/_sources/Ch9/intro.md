# Regression

(Overview of cost functions and regression methods)

## Normal Equations

Finding $w$ that minimizes $||Xw-y||^2$, aka **residual sum of squares**, which can be done by calculus. We apply calculus to find the critical point. 

There exist several ways to compute $\nabla_w ||Xw-y||^2$. First, expanding $||Xw-y||^2$ gives us 

$$w^TX^TXw - 2y^TXw + y^Ty$$

Now, taking the gradient of this with respect to $w$ gives us

$$2X^TXw-2X^Ty$$

When we set this equal to 0, we can reduce this further to give us **normal equations**

$$X^TXw = X^Ty$$

First, think of $X^TX$ as a single $(d+1) \times (d+1)$ matrix (accounting for the bias term). Of course we know $w$ is a d+1-length vector, and  of finally, $X^Ty$ is a d+1-length vector. 

So we multiply a square PSD matrix $X^TX$ by $w$. Of course, $X$ and $y$ are our function inputs, and we want to solve for $w$. 

One thing that comes up: if $X^TX$ is singular, i.e. all sample points lie on a common hyperplane in $d+1$- dimensional space, the normal equations are **underconstrained**. This gives us a situation where $X^TX$ might not be positive definite, and may have some 0 eigenvalues. When this occurs, we know $w$ has more than one solution. 

However, there will always be at least one solution for $w$. 

For now, suppose that $X^TX$ is invertible and positive definite. Then, there will be a unique solution $w^*$. We use a **linear solver** to find $w^* = (X^TX)^{-1}X^Ty$. 

Note that $(X^TX)^{-1}X^T$ is a **linear transformation** of $y$ to weight vector $w$. It is called the **pseudoinverse** of $X$, or $X^+$ for short. **Every matrix $X$ has a pseudoinverse**. Note that it is a d+1 x n matrix (whereas $X$ is an n x d+1 matrix). In an ideal world, if the points $y$ did actually lie on a hyperplane, then $y = Xw$. So it's only natural that we take the inverse of $X$ to get $w$. 

If $X^TX$ is invertible, then $X^+$ is a **left inverse** of $X$. Note: $X^+X = (X^TX)^{-1}X^T = I$, so we see that it is a left inverse. Note that $XX^+$ is generally NOT equal to $I$. 

Once we do the regression, we can go back and look at our predictions for our sample points using the regression function. A prediction for sample point $X_i$ will give us $\hat{y}_i = w \cdot X_i$. Doing it all at once gives $\hat{y} = Xw = XX^+y = Hy$, where $H = XX^+$. $H$ is called the **hat matrix**, which is an $n \times n$ matrix, since it is a linear transformation that puts a hat on $y$. 

So $y$ is the real set of labels, $\hat{y}$ is our predictions. 

## Advantages of Least-Squares Regression (vs. Other Regressions)

Least squares regression is easy to compute for $w$, since we're just solving a linear system. It also gives a unique solution, unless underconstrained (there are still ways to get solutions from here!). It's generally considered a  **stable solution** as well: small changes to the data $X$ will not likely change your solution.

## Disadvantages of Least-Squares Regression 

Least-squares is very sensitive to outliers, since errors are squared. Additionally, if $X^TX$ is singular, then it won't have a unique solution and we need another method to find a valid solution out of many. 

## Logistic Regression

Now, we use the logistic regression function, whose outputs can only be *probabilities*- thus between 0 and 1. 

The main application for logistic regression is classification. Most applications have labels of $y_i$ as 0 or 1. 

Remember that **generative models** build a full probability model of all probabilities involved, i.e. class-conditional distributions. These include LDA and QDA. **Discriminative models**try to interpolate and model the posteriors *directly*. **Posterior probabilities are often well-modeled by the logistic function.** 

So the goal is to find $w$ that minimizes the cost function 

$$J(w) = \sum_{i=1}^{n}L(x \cdot w_i, y_i)$$

where $L$ is the **logistic loss function**. Plugging that in, we get: 

$$J(w) = \sum_{i=1}^{n}-y_i\ln s(X_i\cdot w) + (1-y_i)\ln (1-s(X_i \cdot w))$$

Let's take a look at what exactly we're minimizing. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210316153556472.png" alt="image-20210316153556472" style="zoom: 25%;" />

These are graphs of the logistic function of our prediction given $y$. In the left example, the loss function is 0 when the prediction is 0, matching the true label 0. However, loss goes to infinity when prediction goes further away to 1. It is obvious that the minimal loss occurs when the prediction matches the truth. 

Fortunately, our cost function $J(w)$ is **smooth and convex**. Many ways to solve it, including gradient descent and Newton's method. Let's solve it with GD.

First, we know that the derivative of the sigmoid function $s(\gamma)$ is $s'(\gamma) = s(\gamma)(1-s(\gamma))$. This is its graph:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210316154137417.png" alt="image-20210316154137417" style="zoom:25%;" />

We can see that the derivative is maximized at 0: this shouldn't be surprising, considering the sigmoid function's slope is also maximized at $\gamma = 0$. 

To proceed with gradient descent, we need the gradient of the cost function with respect to $w$. Let $s_i = s(X_i \cdot w)$. 

$$\nabla_wJ(w) = -\sum_{i}(\frac{y_i}{s_i}\nabla s_i - \frac{1-y_i}{1-s_i}\nabla s_i)$$

We know that $\nabla s_i$ is basically the derivative of sigmoid, so we can reduce this to 

$$\nabla_wJ(w) = -\sum_{i}(\frac{y_i}{s_i} \frac{1-y_i}{1-s_i})s_i(1-s_i)X_i$$

$$= -\sum_{i}(y_i - s_i)X_i$$

$$= -X^T(y-s(Xw))$$

where $s(Xw)$ is just a vector of $s(X_i \cdot w)$. 

## Gradient Descent Rule

$$w^{(t+1)} = w^{(t)} + \epsilon X^T(y-s(Xw))$$

## Stochastic Gradient Descent Rule

$$w^{(t+1)} = w^{(t)} + \epsilon (y_i-s(X_iw))X_i$$

Notice here it's basically the same as standard GD, but we're doing steps **one point at a time**. This algorithm works best if we shuffle the points randomly, then process one by one. 

For large $n$, it is common that SGD converges before we process all the points.  

Notice the algorithm's similarity to the perceptron learning rule: 

$$w^{(t+1)} = w^{(t)} + \epsilon(y_i)X_i$$

We just start by setting $w = 0$, and this will always converge. 

If sample points are linearly separable, then logistic regression will always find a separator. Let's say linear separator $w \cdot x = 0$ separates them- the decision boundary doesn't touch any of the points. Let's say we scale $w$ to have infinite length. When this happens, $s(X_i \cdot w) \to 1$ for points in class C, while they go to 0 for points not in class C (correct prediction probabilities maximized). As a result, $J(w) \to 0$. Therefore, logistic regression always finds a separator.