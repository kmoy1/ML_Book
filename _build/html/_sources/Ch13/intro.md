Kernels
========================

WARNING: The topic that is to come is *very* difficult. 

This note focuses on **kernels**, also known as the **kernel trick** or **kernelization**. It's pretty magical- let's see why. 

To motivate kernels, let's recall polynomial features. Say we have a dataset with $d$ features and we want to fit a degree-$p$ polynomial to get a more complicated decision boundary. Of course, overfitting is always an issue here. However, another issue comes where we rapidly blow up the number of features we are using. Specifically, we blow the number of features up to $O(d^p)$ features when we have a degree-$p$ polynomial with $d$ features! For example, say we have $d = 100$ features and we want a degree $p = 4$ decision function- each lifted feature vector needs to account for *all combinations* of features, and thus will have about *four million* features. Not good. 

So in this note we'll show we can still fit these polynomials very fast - we don't need to compute all $d$ features to use them. There's a very clever mathematical trick that allows us to not compute our $d^p$ features, but still get our polynomial anyway! 

Kernelization is based on two observations about many learning algorithms. The first observation: often, when you compute the optimal solution for an optimization problem, you often discover that the **solution can be written as a linear combination of the sample points**. Why this happens is kind of abstract and certainly out of scope, but it does hold true for a lot of algorithms: SVMs, ridge regression, perceptrons, logistic regression,  etc. The second observation: with the right organization of computation, **these algorithms can use inner products of lifted feature vectors only.** For a sample point $x$, our lifted feature vector is $\Phi(x)$. A lot of these algorithms just need to have an inner product of a couple of $\Phi(x)$ vectors as their central computation. With the right conditions, we actually **don't need to know $\Phi(x)$** **to find their inner product**! 

Let's take a mathematical look into observation 1. Let's suppose optimal weight vector $w$ (could be many) takes the form: 

$w = X^Ta = \sum_{i=1}^{n}a_iX_i$

where $X$ is our design matrix, and $a$ is our coefficient vector that tells us what linear combination of sample points give us our optimal solution. If this is true for some $a \in R^n$. Once we know this is possible, we can derive a new form of our algorithm, that's based on finding the optimal $a$ instead of finding the optimal $w$. We do this by simply substituting $w = X^Ta$ into our algorithm and solving for $a$ instead of $w$. The entries of $a$ are called **dual weights**, or sometimes called dual parameters. We thus are optimizing $n$ dual weights instead of $d+1$ (or $d^p$) primal weights $w$. 

Let's use ridge regression as an example- this will lead to a **kernel ridge regression** algorithm. Before we dualize ridge regression, there's one small detail we need to take care of first. Before, in ridge regression we do not have to penalize the bias term- however, in this case we do, because we're assuming that weights are a linear combination of the sample points- this is only true IF we penalize the bias term. One way to minimize damage from this is by first centering $X$ and $y$ (so their means are zero)- for each sample point (row) $X_i$, subtract the mean $\mu_X$. Similarly, subtract each $y_i$ by $\mu_y$. *Don't subtract from the bias column, though.* 

So now that things are centered, it's less harmful to penalize our bias term. This means for ridge regression, instead of $I'$ where the last diagonal element of $I'$ was 0, we can just use the identity matrix $I$ in our normal equations. Recall our normal equations for the standard primal form of ridge regression:

$(X^TX + \lambda I)w = X^Ty$

The reasoning centering helps: if we have data from a random distribution, then the expected linear regression will pass through the origin. So centering $X,y$ will have the decision boundary likely to pass through the origin or close to it. 

Let's now look at the normal equations for the dual form of ridge regression, which we can obtain from above via substitution. Suppose we have a vector $a$ which is a solution to:

$(XX^T+\lambda I)a = y$

Then, $X^Ty = X^TXX^Ta + \lambda X^Ta = (X^TX+\lambda I)X^T a$.

Note that the first term $(X^TX+\lambda I)$ matches the first term in the primal form. Thus, $X^Ta$ *must be a solution for* $w$ in the primal normal equations. Thus, we conclude $w = X^Ta$ is a solution to the primal normal equations. Moreover, we see that $w$ is indeed a linear combination of sample points from $X$! This is key to making kernels and duality work. 

So we call $a$ the **dual solution**. It solves the dual form of ridge regression and we want to find $a$ that minimizes 

$||XX^Ta-y||^2+\lambda||X^Ta||^2$

We got this objective function by simply plugging in $w = X^Ta$ into the original cost function for ridge regression. We can easily verify $a$ by taking the gradient of this function and solving for $a$. 

For the training part of dual ridge regression, we first solve the normal equations for dual weights $a$. Since we know this system of linear equations $(XX^T+\lambda I)$ is symmetric and positive definite, it's easy to solve and has a unique solution. For testing, our regression function on test point $z$ is $h(z) = w^Tz = a^TXz$. Note that $a^TXz$ is also a linear combination, or weighted sum, of inner products:

$a^TXz = \sum_{i=1}^{n}a_iX_i^Tz$

This is key: remember I mentioned earlier that we calculate inner products of lifted feature vectors- the $X_i^Tz$ is our inner product here that will be **very** important for what we want to do later. 

Now that we have a good understanding of dual ridge regression, let's define a little terminology.

First, let us define the **kernel function** $k(x,z)$, where $x$ is. For now, let the kernel function simply be the dot product of its input vectors: $k(x,z) = x^Tz$. Later, we'll incorporate the lifting map $\Phi$ on the two vectors before taking their inner product, but we're not there yet. 

Now, define **kernel matrix** $XX^T$ be an $n \times n$ matrix. Note that $X$ will have the bias dimension here. $K$ is defined such that $K_{ij} = k(X_i, X_j)$. $K$ is always positive semidefinite, but not necessarily positive definite. It is quite common for $K$ to be singular, and this is quite common if $n > d+1$- and singularity can even happen even when this is not the case. 

If this happens, don't expect a solution to your dual ridge regression problem if $\lambda = 0$ (no regularization). This means we probably want *some* regularization- and this is good anyway to reduce overfitting. 

Kernelization and kernel algorithms is most interesting when $d$ is very large, since our lifting map adds a lot of new features anyway. So the  $n > d+1$ isn't something *too* worrisome. 

Now let's write out the dual ridge regression algorithm in a manner that uses the kernel matrix and function, so we can apply it to kernelization. First, we compute the kernel matrix $K$: simply calculate $K_{ij} = k(X_i, X_j)$. Once we have this kernel matrix, we apply it to our dual normal equations to get

$(K + \lambda I)a = y$

which gives us a linear system of equations to solve for dual weights $a$. Again, this is the training portion of the dual algorithm: for testing, we calculate $h(z)$ for each test point $z$, where $h(z) = \sum_{i=1}^{n}a_ik(X_i, z)$. We're going to work some magic to make our kernel function $k$ *supremely fast*. 

But before we do that, let's calculate the runtime of our dual algorithm. First, in calculating $K$, we calculate $n^2$ entries as $k(X_i, X_j)$. Each calculation of $k(X_i, X_j)$ is the dot product which has $O(d)$ runtime, so computing $K$ is $O(n^2d)$. Then, solving our $n \times n$ linear system of equations generally takes $O(n^3)$ time. Finally, for each test point, we compute $h(z)$, which takes $O(nd)$ time. So overall, since we're considering $d >> n$, our dual algorithm takes $O(n^2d + n^3)$ time.   

Note our dual algorithm *does not use sample points $X_i$ directly*. It is only used as an input to our kernel function $k$. So this means if we can configure our kernel function $k$ to avoid using $X_i$'s value directly, this will be great for speed. 

Now let's compare the dual algorithm with the primal for ridge regression. In the dual, we solve an $n \times n$ linear system, and in the primal, we solve a $d \times d$ system. We don't transpose in the primal, so $n$ and $d$ simply swap places- meaning we have a $O(d^3 + d^2n)$ runtime. 

This means that the choice between dual or runtime depends on comparing $d$ and $n$. For raw runtime, we prefer dual when $d > n$, and primal when $d \leq n$. Practically, though, we know that we'll usually have way more sample points than features: $n >> d$. However, remember we're using polynomial features and parabolic lifting, so we're actually gonna have way more features than we think. This is why the dual might be useful here! **Moreover, adding polynomial terms as new features will blow up $d$ in the primal algorithm, but will stay constant in the dual.** 

Finally, remember that **dual and primal produce the same exact predictions**. They are just different ways of doing the same computation. 

## The Magic: Kernelization

Now finally, the magic part. We can compute a polynomial kernel with *many* monomial terms *without actually computing the individual terms itself*. 

The polynomial kernel of degree $p$ is given as $k(x,z) = (x^Tz + 1)^p$. Note that $x^Tz + 1$ is a *scalar*, so taking it to a power is $O(1)$ time. 

A theorem: $(x^Tz + 1)^p = \Phi(x)^T\Phi(z)$ where $\Phi(x)$ contains every monomial in $x$ of degree $p$ (degree 0 to p). For example, let's say we have $d=2$ features, and want a degree $p=2$ polynomial. Let's assume $x, z \in R^2$. Then, 

$k(x,z) = (x^Tz + 1)^2 = x_1^2z_1^2 + x_2^2z_2^2 + 2x_1z_1x_2z_2 + 2x_1z_1 + 2x_2z_2 + 1$.  

We can factor this into an inner product of two vectors, one with $x$ terms and one with $z$ terms:

$= \begin{bmatrix}x_1^2 & x_2^2 & \sqrt{2}x_1x_2 & \sqrt{2}x_1 & \sqrt{2}x_2 1\end{bmatrix} * \begin{bmatrix}z_1^2 & z_2^2 & \sqrt{2}z_1z_2 & \sqrt{2}z_1 & \sqrt{2}z_2 1\end{bmatrix}^T$

Now define $\Phi(x)$ and $\Phi(z)$ as these two respective vectors. Which means we can *finally* compute $k(x,z) = \Phi(x)^T\Phi(z)$ like we dreamed of, and calculate it as a single expression $(x^Tz + 1)^p$ instead of actually computing $\Phi(x)$ or $\Phi(z)$ themselves! It'll take $O(d)$ runtime instead of $O(d^p)$ time (by calculating $d^p$ terms in $\Phi(x)$). 

Now, applying kernelization to ridge regression, the important thing to understand: we take our dual and replace $X_i$ with $\Phi(X_i)$. Now, our kernel function is $k(x,z) = \Phi(x)^T\Phi(z)$.  

