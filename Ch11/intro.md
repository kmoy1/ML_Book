# Regularization

Today we'll talk about various methods called **shrinkage**, ways to regularize weights learned to be smaller than they should be.

The first method is **ridge regression**: it uses L2 regularization (sometimes called Tikhonov regularization). It's a lot like OLS, but we use the $l2$ penalized mean loss. So we find $w$ that minimizes 

$$||X \cdot w - y||^2 + \lambda||w'||^2$$. 

The regularization term (aka penalty term) takes $w'$: it is the same as $w$ except the bias term is 0 (we don't penalize the bias term). This regularization term "encourages" the weights in $w$ to be small, as we penalize larger ones. By shrinkage of $w$, we ensure our normal vectors are not too long. 

Two common reasons why we want shrinkage. First of all, **ridge regression guarantees a positive definite matrix of normal equations** (whereas only positive semidefinite was guaranteed before). For example, if points did not fully span our feature span, i.e. they were on a hyperplane of feature space, then least-squares linear regression DOES NOT have a unique solution, since our matrix is PSD (might have a few 0-eigenvalues). **This ensures there's always a unique solution**. 

Take a look at the diagram below:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315134352411.png" alt="image-20210315134352411" style="zoom:33%;" />

On the left, the quadratic form from a least squares regression problem. `beta1, beta2` are the feature space. The z axis is the cost function we want to optimize. Note there's more than one solution, indicated by the flat line at the bottom of the paraboloid. Such a regression problem is said to be **ill-posed**. 

Now on the right, adding the regularization term turns our ill-posed problem into a **well-posed** problem- there's one unique solution now. 

So the second motivation: we reduce overfitting by reducing variance. Imagine regression gives us a curve $h(x) = 500x_1 - 500x_2$. However, the points are well-separated. But the labels are constrained as $y_i \in [0,1]$. So the points aren't close, but the labels are. The weights being that large doesn't really make sense- a very small change in $x$ can give a very large change in $y$! When a curve $h$ oscillates a lot, it's a sign of overfitting and high variance. 

Recall our objective function $J(w) = ||Xw -y||^2$ has weights $\beta_1, \beta_2$. At the center of the contours of $J(w)$, we have our optimal solution WITHOUT regularization.

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315135520811.png" alt="image-20210315135520811" style="zoom:33%;" />

However, once we do add regularization term $\lambda||w'||^2$, the isocontours of this term are perfect spheres (blue circle above). What we want to do is minimize a weighted combination of regularization and squared error. **The optimal solution is where an isocontour of 1 barely touches an isocontour of the other**. Note the point where the red touches the sphere: that is ONE solution for a particular choice of $\lambda$. But for different $\lambda$, it'll be different. For all possible solutions given all possible lambda, we get a *curve* of all optimal solutions. One endpoint is when $\lambda$ = 0: this is the center point of the red ellipses.

We can still solve $\nabla J(w) = 0$  through calculus. Specifically, we get 

$$(X^TX + \lambda I')w = X^Ty$$

where $I'$ is identity matrix with the bottom right term set to 0 (since we don't penalize the bias term).

Once we solve for $w$, we just return the hypothesis as $h(z) = w^Tz$.

With increasing lambda, we of course apply more regularization, forcing $||w'||$ to get smaller and smaller. Assuming a model of the data where we assume the data came from a linear relationship with Gaussian noise ($y = Xv + e$), the variance of ridge regression is equal to 

$$\text{Var}(z^T(X^TX+\lambda I')^{-1}X^Te)$$

which is the standard variance over the distribution of all possible $X,y$. The matrix $\lambda I'$ gets larger as lambda does, so as $\lambda \to \infty$ variance goes to 0, BUT bias will increase. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315140946841.png" alt="image-20210315140946841" style="zoom:33%;" /> 

As lambda gets larger, variance drops to 0, but bias (squared) is increasing. In fact, the hypothesis ITSELF is actually pushed to 0 (since its weights are), so once it gets super small it's obviously not a good approximation of $g$. The point X where we minimize test error is considered optimal and the lambda we want. 

Note the regularization term $\lambda ||w'||^2$ penalizes all weights equally- this isn't generally true in practice. For that reason, it makes sense to *normalize* all features s.t. they all have the same variance, and are thus on the same "scale". 

Alternatively, we can use an *asymmetric penalty* by replacing $I'$ with some other diagonal matrix. For example, cubic features should not have the same penalty as linear features. 

## Bayesian Justification for Ridge Regression

We can justify using an L2 regularization term in ridge regression.

First, we assign a prior probability on $w'$: we don't assume all weights are equally likely. If a weight is big, be skeptical, if small, trust more. We can express this by stating that our weights $w' \sim N(0, \sigma^2)$. So very large $w'$ will have very low PDF values (probabilities) in the normal distribution.

Now, we **apply MLE to the posterior probability**. Remember Bayes theorem gives posterior 

$$f(w|X, y) = \frac{f(y|X, w) * f(w')}{f(y|X)}$$

where $f(w')$ is our prior, normally distributed.  

We can also think of the class-conditional probability/density as the likelihood function of $w$ given $X, y$, so we get 

$$f(w|X,y) = \frac{L(w) * f(w')}{f(y|X)}$$

Now maximizing the log posterior gives 

$$ln(L(w)) + ln(f(w')) - C = -C_1||Xw-y||^2-C_2||w'||^2-C_3$$ 

which finally leads to MINIMIZING

$$||Xw-y||^2 + \lambda||w'||^2$$.

So the process is:

1. Assuming weights follow a distribution- some more likely than others.
2. Write expression for posterior probability
3. Apply MLE to posterior to get ridge regression function. 

We find the value of $\lambda$ by validation. 

## Feature Subset Selection

Now what if the goal was to get rid of features that weren't very predictive? When there's a shitload of features, there comes a point where most just increase variance without reducing bias. 

So the idea is we identify the poorly predictive features and effectively get rid of them by setting corresponding weights to 0. This means less overfitting and smaller test errors. Another motivation is *inference*: we apply some human-understandable rules as features. The simpler your model, the easier to interpret it meaningfully.

Pretty much all classification and regression benefits from this idea. But this could be a difficult problem. Different features can encode the same information in different ways. So it can be hard to figure out which subset is the best subset. 

The first algorithm for best subset selection is to just try all $2^d-1$ nonempty feature subsets, and train a classifier for EACH. We choose the best one by validation. Of course, this is incredibly inefficient and slow if $d$  is large. 

So here's where heuristics come into play.

## Heuristic 1: Forward Stepwise Selection

Forward stepwise selection adds one feature at a time to the model. We start with the null model (0 features). Then, add the best feature  left at each iteration. To select the best feature, we just train $d$ models and compare with validation. We stop when validation errors start to increase instead of decrease (from overfitting).

So now we're training $O(d^2)$ models instead of $O(2^d)$. Better.

But not perfect: for example, there may be cases where we have a 2-feature model if neither of the features yield the best 1-feature model (they have to be good individually).

## Heuristic 2: Backward Stepwise Selection

This is now pretty straightforward (straightbackward). Now, we start with all $d$ features and remove one at a time until validation error starts to decrease. 

This also trains $O(d^2)$. 

Which is the better choice? Depends on how many features you think will be useful. If you only think a few features would be good, go forward, and vice versa. For example, for spam classification FSS is probably better. 

## Lasso Regularization

Now we move on to L1 regularization- L1 penalized mean loss instead of L2.  

Lasso, incidentally, is an acronym: for "least absolute shrinkage + selection operator". 

The advantage of L1 regularization over L2 is it **naturally sets some weights to zero.** This can be really useful in some cases, at it sort of emulates subset selection. However, it's harder to optimize (minimize) our objective function:

$$||Xw-y||^2 + \lambda||w'||_1$$

where $||w'|| = \sum_{i=1}^{d}|w_i|$ : the sum over the $d$ components of the normal vector. Again, we don't penalize the bias term. 

Recall the isosurfaces of $||w'||^2$ in ridge regression are hyperspheres. However, the isosurfaces of $||w'||_1$ are **cross-polytopes**. The unit cross-polytope is the convex hull of all unit coordinate vectors, including positive and negative versions of those vectors. 

For example, the convex hull of a 2D vector (where axes are the +/- of each coordinate vector) is this diamond shape:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315144632065.png" alt="image-20210315144632065" style="zoom:50%;" />

This diamond shape is the isocontour (isosurface) where $||w'||_1 = 1$. 

In 3D, you can imagine the isosurface looks like this: 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315144755070.png" alt="image-20210315144755070" style="zoom:50%;" />

Note for $d$ dimensions, we'll have $2d$ coordinate vectors, and taking the convex hull of them gives the polytope in $d$-dimensional space.

What does this do in regularization? Remember in ridge regression, we find the point where the ellipses of regression and sphere of regularization just barely touch. This was an optimum for SOME value of $\lambda$. It's the same thing with L1, but now instead of a sphere, the isosurfaces are now convex hulls. 

One of the solutions in LASSO will have one of its weights set to 0. **The bigger $\lambda$ is, the higher tendency to set weights to 0**. 

When a regularization ellipse touches a TIP of the convex hull, then ALL weights get set to 0 EXCEPT for 1. 

Let's look at a graph that shows how weights behave for a higher dimensional space. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315145511201.png" alt="image-20210315145511201" style="zoom: 50%;" />

This chart shows that as $\lambda$ increases, we see the evolution of different weight variables. The chart indicates that four weight curves, and thus four weights, are truly notable while 6 are not. 

Two main algorithms for solving LASSO are called least-angle regression (LARS) and forward stagewise.

Again, like ridge, we probably want to normalize features for LASSO. 

