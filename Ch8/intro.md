Anisotropic Gaussians
==========================

Today, we'll take a look at **anisotropic Gaussians**: normal distributions where its isocontours are *not* perfect spheres (ellipsoidal). Another way to put this: the standard deviation is longer in certain directions. In this lecture, we'll look at the consequences of this for LDA and QDA. Different directions, different variances. 

Recall the multivariate normal PDF is: 

$$f(x) = n(q(x))$$

where $q(x) = (x-\mu)^T\Sigma^{-1}(x-\mu)$ is the quadratic part of $f$ mapping a d-dimensional vector to a scalar value, and $n(q)$ is the outer part that is an *exponential map* that maps a scalar to another scalar, in this way: 

$$n(q) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}}e^{-q/2}$$.

Remember that $\Sigma$ is the covariance matrix of the normal distribution, and $|\Sigma|$ is the *determinant* of the covariance matrix. 

$\Sigma, \Sigma^{1/2}, \Sigma^{-1}$ all play a role in intuition about the multivariate normal distribution. The *eigendecomposition* of each one of these tell us about more of these.

Let's start with the eigendecomposition of $\Sigma = V\Lambda V^T$, where $V$ is a square matrix with eigenvector columns, and $\Lambda$ is a diagonal matrix of $\Sigma$'s eigenvalues. In one dimension, $\sigma^2$ represents variance. In multiple dimensions, like this case, the **eigenvalues of the covariance matrix are the variances along the eigenvectors.** So we can label an eigenvalue $\Lambda_{ii} = \sigma_i^2$. 

If we want to think in terms of the width(s) of the multivariate Gaussian itself, we look at $\Sigma^{1/2}$, where the eigenvalues are the square roots of the variances, and the eigenvectors stay the same. The special thing about $\Sigma^{1/2}$ is that it **maps spherical isocontours to ellipsoidal isocontours** (of the normal PDF). $\Sigma^{1/2}$ plays the role of $A$, discussed last lecture. The **eigenvalues of $\Sigma^{1/2}$ are the widths/standard deviations of the ellipsoids:** $\sqrt{\Lambda_{ii}} = \sigma_i$. 

Let's look at the linear transformation being done here visually. We start with a quadratic function (basically $z^Tz = ||z||^2$) on the left, which has spherical isocontours. But the linear transformation $\Sigma^{1/2}$ gets the isocontours of $q(x) = (x-\mu)^T\Sigma^{-1}(x-\mu)$, which is the **quadratic form** of $\Sigma^{-1}$. 

**Isocontours of **$q(x)$ will also be isocontours of the Gaussian PDF. 

Lastly, $\Sigma^{-1}$ has eigenvalues $\frac{1}{\Lambda_{ii}}$. $\Sigma^{-1}$ is called the **precision matrix**. It is the quadratic form of the precision matrix that give the isocontours that make the normal distribution PDF. 

The second step in this process: once we understand the geometry of the quadratic form of the precision matrix, we run it through $n$, which is a **smooth and convex function**. After this, we get the shape of the Gaussian PDF! The isocontours are the same, but the **isovalues have changed**. In particular, $x$ that minimized $q(x)$ now maximizes the Gaussian PDF. For a weight vector, as we get infinitely far from the mean, $q(x)$ goes to infinity, but the Gaussian PDF goes to 0. 

## MLE for Anisotropic Gaussian Distribution

Remember the premise of MLE: we are trying to estimate the parameters that maximize the likelihood that we see the observed data from our distribution. Given sample points $X_1, ..., X_n$, and class labels $y_1, ... , y_n$, we want to fit Gaussians **to each class** this time. We express our sample points as column vectors to fit the PDF definition. 

For QDA, we estimate the covariance matrix for class C as: 

$$\hat{\Sigma}_C = \frac{1}{n_C} \sum_{i:y_i=c}(X_i-\hat{\mu}_c)(X_i-\hat{\mu}_c)^T$$

This is called the **conditional covariance** for points in class C. For each sample point in C, we sum an **outer product matrix**: specifically, subtract the sample mean and take the outer product of that vector with itself. We sum this over all the points in class $C$, then divide by the number of points in class $C$. 

For QDA, we calculate $\hat{\Sigma}_C$ for each class C, as well as priors and means as usual. Once we have these parameters, we have the Gaussian PDF that is best fit to our sample points $X_i$. Then, once we have the distributions, doing QDA/LDA is straightforward. 

Note that $\hat{\Sigma}_C$ is a **sample covariance matrix**: it is always positive semidefinite, but not necessarily positive definite (can have some eigenvalues of 0, which give it 0-length directions). **If it's not positive definite, it doesn't actually define a true normal distribution.** This will result in linear isocontours instead of ellipsoidal!

What about LDA? We know that $\Sigma$ is constant for all classes. The sample covariance matrix in LDA, then, is:

$$\hat{\Sigma} = \frac{1}{n}\sum_c \sum_{i:y_i=c}(X_i-\hat{\mu}_c)(X_i-\hat{\mu}_c)^T$$

So what we're basically doing is taking a weighted average of the covariance matrices for each class, weighted by the number of points in that class. This is called the **pooled within-class covariance matrix**. The "pooled" comes from the fact that we're summing over all classes. **The usage of within-class means $\mu_C$ tend to make covariance smaller.** Once we figure out this matrix, we just use it for each class. 

## QDA

So now that we've used MLE to get our best-fit Gaussians, now comes prediction. 

The big idea, of course, is to choose class $C$ that maximizes the posterior, or maximizes $f(X=x|Y=C)\pi_C$ : the product of the class-conditional density and prior. In QDA, this is equivalent to maximizing the **quadratic discriminant function** $Q_C(x)$: 

$$Q_C(x) = \ln((\sqrt{2\pi})^df_c(x)\pi_c)$$

$$= -\frac{1}{2}(x-\hat{\mu_c})^T\Sigma^{-1}(x-\hat{\mu_c})-\frac{1}{2}\ln|\Sigma_C|+\ln\pi_C$$

So we just compute for each class and pick the one with the highest discriminant function. 

In the two class case, it's even simpler: our decision function is $Q_C(x) - Q_D(x)$ which is quadratic. However, this may be **indefinite because we are subtracting 2 PSD matrices**. The Bayes decision boundary will always be a solution to a multivariate quadratic. We can find the posterior probability $P(Y|X) = s(Q_C(x) - Q_D(x))$. 

Let's take a look at this graphically. Below is a graph of the two PDFs that we fit to our classes, $f_C(x)$ and $f_D(x)$ (x,y axes are features, z axis is PDF value).

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210317175227382.png" alt="image-20210317175227382" style="zoom: 50%;" />

This actually has a **hyperbolic decision boundary**- it is possible since quadrics in 2D are conic sections. This means that the set of points in predicting a class isn't connected- it's in 2 disjoint regions!

Now, if we plot the decision function $Q_C(x) - Q_D(x)$, we see it is indeed a hyperplane: 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210317175412116.png" alt="image-20210317175412116" style="zoom:33%;" />  

The decision boundary is much clearer here: we don't need the Gaussian PDFs to find it here. Also, since it's quadratic, the boundary can be computed MUCH more quickly. 

If we want the posterior probability, i.e. the probability our prediction is correct, we just pass our decision function through the sigmoid function as $s(Q_C(x) - Q_D(x))$. 



