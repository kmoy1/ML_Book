# Statistical Justifications For Regression

Last 2 lectures explained ways to fit curves to points. Early on, I divided ML into 4 levels: application, mathematical modeling, optimization, optimization algorithm. Last 2 lectures talked about optimization. 

I want to start by having a model *produces* the real world data that we want to analyze. Assume sample points all come from some unknown probability distribution: $X_i \sim D$. Each sample point has a (numerical) label, each with noise. Specifically, y values are a sum of an unknown function that we want to APPROXIMATE, and errors induced by real-world measurement: $y_i = g(X_i) + \epsilon_i$ for some unknown "grounds-truth" function $g$ and errors $\epsilon_i$. We are going to assume $\epsilon \sim D'$, where we make a few big assumption: $D'$ **has mean 0**, and that the errors are INDEPENDENT from $X$ themselves. Our model also leaves out **systematic error**: your measurement device always adds 0.1 to every measurement. 

Now that we have a model for where our data comes from, we want to do regression on that data. The goal is to find a **hypothesis function** $h$ that is an accurate estimate of the unknown ground truth $g$. Formally, we want to choose $h(x)$ as the expected value over all possible labels that we might get for our random data point. Formally, $h(x) = E_Y[Y|X=x] = g(x) + E[\epsilon] = g(x)$. If the expectation $E_Y[Y|X=x]$ exists at all, then it partly justifies our model of reality. We can retroactively define ground truth $g$ to *be* this expected value.

Another way to think about this is graphically. Given feature space on the x axis and $g(x)$ on the y axis, let $x_1$ be a point where we take a measurement. If we had no noise, then $g(x_1)$ would be our measurement, but our actual measurement is $y_1$, with error coming from a Gaussian distribution. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210315115140571.png" alt="image-20210315115140571" style="zoom:33%;" />

With many samples, we could estimate the mean of this error Gaussian. We can also define $g(x_1) = E_Y[y_1]$. 

Now that we have this model of reality, we're going to first apply it to justify least squares regression. First, why does least squares look at a sum of squared errors instead of just the sum of absolute errors? If you have this normally-distributed noise, by applying *maximum likelihood estimation*, we actually *prove* least squares is the right way to do (linear) regression. Let's prove this below. 

Suppose $\epsilon_i \sim N(0, \sigma^2)$. This means the label $y_i$ also comes from a normal distribution: $y_i \sim N(g(x_i), \sigma^2)$. Now let's apply MLE to guess what parameter $g$ is most likely to fit. 

Recall: MLE was used before to guess parameters of a distribution, GIVEN observed data. Now we'll just it to guess parameters of a *function* $g(x)$. 

Remember that the log of the normal PDF $ln(f(y_i))$ is:

$$ln(f(y_i)) = -\frac{(y_i-\mu)^2}{2\sigma^2} - \text{constant}$$

The only part that's relevant here is the quadratic first term. $\mu$ is the mean of distribution, i.e. $g(x_i)$ in this case. Now, the log-likelihood of getting the particular sample points and labels is denoted as $l(g; X,y)$. Notice that $g$ is a function, but is treated as a parameter in this case. We know $X,y$ are given. 

Of course, we know the log-likelihood is the log of the products of PDFs for each sample point. This turns into a sum:

$$l(g; X, y) = \sum_{i=1}^{n}ln(f(y_i)) = \frac{-1}{2\sigma^2}\sum_{i}(y_i - g(x_i))^2 - C$$

where $C$ is a constant which we don't really care about. Once again, we want $g$ such that $l(g; X, y)$ is maximized. It turns out, actually, that maximizing this expression is equivalent to *minimizing* $\sum_{i}(y_i - g(x_i))^2$: the sum of squared errors. This is least squares! 

The big takeaway: if we do MLE on parameter-function $g$, it is equivalent to estimating $g$ by least-squares regression. So if the noise is normally distributed, MLE justifies using the least-squares cost function. Specifically, least squares is there *because* we assume noise is Gaussian distributed. 

Remember that least squares does have a problem though: it's very sensitive to outliers. If noise really is normally distributed, outliers aren't a big deal because they're not that common, and making $n$ large enough compensates. In the real world though, noise isn't usually normally distributed (e.g. heavy-tailed distribution). 

## Empirical Risk 

Remember: **risk is expected loss**. In regression, we want the risk of our hypothesis function. The risk here is the expected value of the loss $R(h) = E[L(h(x), y)]$ over all $x \in R^d, y \in R$. 

Now if we have a discriminative model, e.g. logistic regression, then we don't actually know $X$'s distribution $D$. So how can we minimize risk? Note that if we had a *generative* model (GDA), we could estimate joint probability distribution for $X,y$ and explicitly derive the risk. However, most of the time this distribution doesn't exist/ we don't know. 

So what we do instead: pretend sample points *are* the distribution- the **empirical distribution**. This is the uniform distribution over the sample points. Note that this is a **discrete uniform distribution**, over sample points that exist. Each sample point has equal prior $\frac{1}{n}$. Then, the expected loss for that distribution is called **empirical risk** $\hat{R}(h) = \frac{1}{n}\sum_{i=1}^{n}L(h(x_i), y_i)$. Often this is the best we can do, and good news: often with enough sample points, the empirical risk converges to the true risk. 

So finding the $h$ that *minimizes* the empirical risk $\hat{R}(h)$ is empirical risk minimization. This is a core part of machine learning. This is why we usually minimize a sum of loss functions.

## Logistic Loss from MLE 

We know the (log) logistic loss function is  

$$\sum_{i}-y_i\log(s_i) - (1-y_i)log(1-s_i)$$. 

Where does this come from? 

Well, first, let's answer the following question: what cost function should we choose for probabilities? Specifically, what loss comes from predicting probabilities? 

Let's set $y_i$ = actual probability that $X_i$ is in class $C$. Suppose we do logistic regression, and it gives us prediction $h(x_i)$ after learning $h$. We want to compare $h(x_i)$ and $y_i$ and have a measure of closeness. 

Imagine we run the experiment $\beta$ times, giving us $\beta$ duplicate copies of $X_i$. Then, as $\beta \to \infty$, $y_i\beta$ points *are* in class C, and $(1-y_i)\beta$ sample points are NOT in class C. Note this is a Bernoulli model- but the same result could be reached with Binomial. Now, we'll use MLE to choose weights to maximize probability of getting this sequence. 

We want to find $h$ that maximizes $L(h; X, y) = \prod_{i=1}^{n}h(X_i)^{y_i\beta}(1-h(X_i))^{(1-y_i)\beta}$.  This is representative of the probability of the $y_i\beta$ points in class C and those that aren't. The log-likelihood is $l(h) = \beta\sum_{i}(y_iln(h(x_i)) + (1-y_i)ln(1-h(x_i)))$. Note this is equal to:

$$l(h) = -\beta\sum_{i}\text{logistic loss fn } L(h(X_i), y_i)$$

Again, this is equivalent to *minimizing* the logistic loss function $\sum_{i}L(h(X_i), y_i)$. 

So MLE explains where the logistic loss function comes from, and why we want to minimize the sum of all logistic losses. 

## Bias-Variance Decomposition

There are two sources of error in a hypothesis when doing regression, and in any classification algorithm. These are called the bias and variance. 

The **bias** is the error related to the inability to fit our hypothesis $h(x)$ to the ground truth $g(x)$ perfectly. For example, if $g$ is quadratic, and we try to use a linear $h$, it won't be a good approximation, and bias is high. 

The **variance** is error related to fitting to random noise in the data. For example, suppose $g$ is linear and we fit it with a linear $h$. Yet $h \neq g$, because of noise in the data. Remember $X_i \sim D, \epsilon_i \sim D'$, and $y_i = g(X_i) + \epsilon_i$. We try to fit $h$ to $X,y$. Because both $X,y$ are random, $h$ will be random as well. Now $h$ is a random variable: its weights are random since its inputs are random. 

Consider an arbitrary *test point* $z$ in feature space ($z \in R^d$)- not necessarily a sample point. Let's say we made a measurement at point $z$, which gives it a label. So we set $\gamma = g(z) + \epsilon$. We really want to consider ALL $z$ in our feature space. On the other hand, our label $\gamma$ is random because it has random added noise. Now, $E[\gamma] = g(z)$, and $Var(\gamma) = Var(\epsilon)$. 

Now let's look at the risk function for this model. Assume loss is squared error. This risk gets *decomposed* into bias and variance. The risk is the expected loss: $R(h) = E[L(h(z), \gamma)]$. The expectation is over some probability distribution- over ALL possible training sets $X,y$... AND over all possible values of test label $\gamma$. Remember $h$ comes from a *probability distribution of hypotheses.* So the weights we get are distributed based on $X,y$. 

Substituting $L(h(z), y) = (h(z) - y)^2$ and simplifying, we get: 

$$R(h) = E[h(z)^2] + E[\gamma^2] - 2E[\gamma h(z)]$$

Note that $\gamma, h(z)$ are independent: $h(z)$ only depends on training data $X$, while $\gamma$ only depends on test point $z$'s label.

Finally, we get this beautiful equation, called the **bias-variance decomposition of risk**: 

$$R(h) = (E[h(z)]-g(z))^2 + Var(h(z)) + Var(\epsilon)$$

The first term $(E[h(z)]-g(z))^2$ is the squared bias of the regression method. The second term $Var(h(z))$ is the variance of the regression method. Finally, the third term $Var(\epsilon)$ is the irreducible error.

## Consequences of Bias-Variance Decomposition

Now, we can formally state underfitting and overfitting. **Underfitting** occurs when there's just too much bias. **Overfitting** occurs when there's too much variance. 

Training error reflects bias, but NOT variance. Test error reflects both, but it's where variance rears its head. 

What happens to bias and variance as you get a bigger and bigger sample? As $n \to \infty$, variance goes to 0 in many distributions. Often this is true for bias as well, *if* your class of hypotheses is rich enough. In other words, if $h$ can fit $g$ exactly, bias goes to 0 as well. However, it is common that $h$ just cannot fit $g$ well enough- so the bias will be large (at most points). 

Adding a *good* feature reduces bias. A good feature has *predictive power* that adds to the features we already have. However, adding a bad feature (say, a random number) is bad but it's rare that bias increases. Adding a feature, good or not, always increases variance. So we want to add features that **reduce bias more than increases variance**. 

Note irreducible error is inherent to measurements in the test set- error that more data cannot help. Note noise in the test set only affects irreducible error- it cannot affect bias/variance of our method. Conversely, noise in the training set only affects bias/variance and not irreducible error. 

Now, bias and variance aren't something we can precisely measure. However, we can use *synthetic data* with specific known probability distributions, then we can compute bias and variance exactly. This gives us a way to test learning algorithms. 

