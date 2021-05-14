Machine Learning Glossary
==================================

Below you'll find a machine learning glossary for the plethora of terms in ML.

## AdaBoost

__AdaBoost__ is a popular [boosting](#boosting) algorithm, and stands for "adaptive boosting". The Big Idea: AdaBoost iteratively learns from the mistakes of [weak classifiers](#weak-classifier) and turns them into strong ones.

AdaBoost is generally used with a [random forest](#random-forest) of [decision stumps](#decision-stumps). Sample points are weighted such that different points are given different degrees of importance to be correctly classified.  

Each base learner is also given a "voting power weight"- a weight that indicates what percent of the vote a base classifier should get. At the end, each classifier we learn will be weighted by how good it was on prediction, and our metalearner prediction is a weighted sum of base learner predictions. Note that the metalearner also gives a posterior probability estimate.

See the [AdaBoost chapter](../Ch20/intro.md) for details.

## Backpropagation

Backpropagation is a dynamic programming algorithm for training neural networks (SGD) in runtime linear to the number of weights. 

## Gradient Descent

A way to find the optimal weight vector for an (ideally convex) cost function $J(w)$ if a closed-form solution does not exist.

### Batch GD

Formula:

$$
w^{(t+1)} = w^{(t)} - \alpha \nabla_w J(w)
$$

We do this for some pre-specified number of epochs. We run through the entire dataset on each epoch.

### Stochastic GD

Like batch GD, but we first shuffle the dataset, and run through the formula for each _sample point_. This way, it is possible to converge in less than an epoch.

## Bias

For a machine learning model, bias is error that results from a model simply being inherently wrong. Models with high bias will miss relevant relationships in a dataset. This will rear its head in both training and testing accuracy. A model with high bias is said to underfit. 

Generally, if a dataset has complex relationships but a model is just too simple, the chances are that the model will have high bias. For example, if a decision boundary should be quadratic but the model is linear, it just won't be able to model the relationship well.

## Boosting

The __Boosting__ algorithm is a special type of ensemble learning algorithm that tries to build a strong model by correcting the mistakes of several weaker ones.

### The Boosting Algorithm

First, create an initial model, and try running it on training data. Error should be trash. On the next iteration, we create a second improved model by reducing errors from the previous one. We do this sequentially until the training data is predicted perfectly or we reach some arbitrary threshold.

## Centering

A centered data matrix has each of its features subtracted from its mean. Each centered feature now has mean 0. We represent a centered data matrix as $\dot{X}$. Centering is applied to data in cases where features are not all on the same scale (measurement units).

Centering changes the intercept, but will not change regression weights for features. Graphically, it is just a translation of data points such that the origin is in the middle.

## Decision Tree

Decision trees are ML models that predicts the value of a target variable based on several input variables. In classification, it splits the feature space into axis-aligned boxes, where regions correspond to 1 or 0. 

Tree construction algorithms are usually top-down: at each step, we choose a split feature and corresponding split point (in that feature) that best splits the set of points we have left. The "best" split point is measured by minimum Gini Index or maximum Information Gain.

Running Times:
- Classifying a test point takes $O(\text{tree depth})$. This is _usually_ $O(\log n)$, but not always.
- For binary features, each node tests $O(d)$ splits. For quantitative features, we must try $O(n'd)$ possible splits, where $n'$ is the points _remaining_ in that node.
- The runtime for building a tree is $O(nd \cdot \text{tree depth})$.

### Entropy

Entropy refers to the uncertainty/disorder in a set of points' possible outcomes. Given a set $S$ of training points with $k$ unique classes, the entropy of $S$ is given by:

$$H(S) = -\sum_{i=1}^{k}p_i \log_2 p_i$$

where $p_i$ is the fraction of items in $S$ that belong to class $i$.

Entropy is maximized when there's an equal split between classes, and minimized when the split is skewed more towards one class. 

### Information Gain

In building a decision tree, we want the split that maximizes the information gain: entropy in parent node - weighted sum of entropy in child nodes. Conceptually, we want to decrease entropy as we move down the tree, so this is equivalent to choosing a split that minimizes entropy in child nodes. There will always exist a split with positive information gain UNLESS:
- (Trivial) a child node has no training points. The parent node becomes a leaf node anyway.
- Ratios of classes in the child node are equal.

### Multivariate Splits

A multivariate decision tree has nodes with multivariate splits: they apply a function to some set of features (instead of just one) and compare splits. This allows for oblique decision boundaries: slanted ones. 

## Fictitious Dimension

The fictitious dimension, or bias term, is a column of 1s added to a data matrix. Without it, all decision functions will be forced to pass through the origin. This can only increase bias, and it is rare to find examples of why you wouldn't want to include a bias term. Generally, the bias term is an indicator of the value that datapoints are centered around.

The inclusion of the fictitious dimension also ensures the residual vector $e$ has mean 0.

For neural networks, the bias term is a constant node added to each hidden layer input as well as the output layer input. The allows shifting the activation function to the left or right for each hidden layer node.

## Class-Conditional Distribution

A class-conditional density models the distribution of observations $X$ _conditioned on_ the fact that it belongs in class $Y=k$. Commonly denoted as $f_k(x)$ or $P(X=x|Y=k)$.


## Dimensionality Reduction

Dimensionality reduction is the transformation of data from a high-dimensional space to a low-dimensional one. Ideally, we would reduce the dimensions to the minimal number of variables needed to represent the data (often called _instrinsic dimension_). 

DR can be used for feature selection, but is much more commonly used in feature projection, where we transform data into a lower-dimensional space. Common techniques for this include:
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Generalized Discriminant Analysis (GDA)

## Decision Stumps

Decision stumps are one-level decision trees. They consist of the root node connected to two leaves. In effect, stumps only consider a single input feature before making a prediction.

A very popular type of boosting algorithm is [AdaBoost](#adaboost).

## Eigendecomposition

The eigendecomposition (sometimes called spectral decomposition) is the factorization of a matrix $X$ into matrices of its eigenvectors and eigenvalues. If $X$ is a square $n \times n$ matrix with $n$ linearly independent eigenvectors (no zero eigenvalues), then $X = V \Lambda V^{-1}$, where columns $i$ of $V$ corresponds to eigenvector $i$ of $X$ (ordered), and $\Lambda$ is a diagonal matrix with diagonal elements as corresponding eigenvalues.

If $X$ is real and symmetric square matrix, then it will have $n$ real eigenvalues as well as $n$ real orthonormal eigenvectors. Then we can represent $X = V\Lambda V^T$.

## K-Nearest Neighbors (k-NN)

k-nearest neighbors is a prediction model where for a given input point, we look at the $k$ closest training examples in the training set. If used for classification, we take the majority class of those neighbors. If regression, we average the target value of those neighbors.

Since the algorithm relies on distance, normalizing features is generally a very good idea (if features have different units). 

$k$ is a hyperparameter that is tuned based on the data. Generally, larger $k$ reduces [variance](#variance), and makes the decision boundary look smoother. However, larger $k$ generally also increases [bias](#bias) as it makes boundaries between classes less distinct. Accordingly, smaller $k$ decreases bias but makes decision boundaries fit to tighter clusters of points, so variance increases.

## Kernel Trick

We can use the kernel trick when we want to compute a solution in a higher dimensional polynomial space, but it is too expensive to calculate lifted feature vector $\Phi(x)$. 

A kernel function $k(x,y)$ can represent high-dimensional (lifted) data $\Phi(x)$ by only operating in the original (non-lifted) data dimensions. The big idea: we don't compute $\Phi(x)$ explicitly. Instead, we represent $X$ as a __kernel matrix__ $K$ where $K_{ij} = k(X_i, X_j)$. The kernel function computes dot products of _lifted_ feature vectors: $k(X_i, X_j) = \Phi(X_i)^T\Phi(X_j)$. For a kernel with $d$ higher-dimensional features, we calculate kernel function as $k(x,y) = \Phi(x)^T\Phi(y) = (x^Ty + 1)^d$.

To apply the kernel trick to a non-linearly-separable $X$, for a lift from $d$ dimensions to $D > d$ dimensions, we solve the _dual problem_. If we _dualize_ by setting $w$ to a linear combination of sample points, i.e. $w = \Phi(X)^Ta$, where $a$ is a length-$n$ vector, then suddenly, $\Phi(X_i) \cdot w$ becomes equivalent to $(Ka)_i$. Now, we solve for the optimal $n$ weights in $a$ instead of optimal $d$ weights in $w$. Optomizing $a$- specifically $a_i$ (one weight of $a$ at a time)- during gradient descent now takes $O(1)$ time. Calculating test point predictions takes $O(nd)$ time. 

Normally, computing $\Phi(x)^T\Phi(y)$ via calculating $\Phi(x)$ and $\Phi(y)$ explcitly would take $O(d^p)$ time, where $p$ is the degree polynomial that $\Phi$ lifts feature vectors up to. With the kernel function $k = (x^Ty+1)^p$, this only takes $O(d)$ time. 

Now predictions are a linear combination of kernel outputs, coefficients determined by $a$. 

Note it is important to center data in $X$ before applying kernelization.

### Primal Weights

In standard optomization, we find $d$ primal weights for weight vector $w$. If we want to fit a degree-$p$ polynomial, we will have to optomize $O(d^p)$ weights. Predictions involve calculating $\Phi(X_i)^Tw$.

We solve a $d \times d$ linear system in the primal algorithm.

### Dual Weights

In kernelized optomization, we find $n$ dual weights for weight vector $a$. Predictions involve calculating a linear combination of kernel function outputs over all training points: $\sum_{j=1}^{n}a_jk(X_j, z)$ for a test point $z$.

We solve an $n \times n$ linear system in the dual algorithm.

For example, in ridge regression, we can find dual solution $a \in \mathbb{R}^n$ that minimizes $||XX^Ta-y||^2 + \lambda||X^Ta||^2$.

## K-Means Clustering

K-means clustering is a way to cluster $n$ observations, by iteratively assigning each observation to the nearest _cluster mean_, then reassigning cluster means, then repeating until convergence. We use Euclidean distance.

## K-Medioids Clustering

K-medioids clustering is like k-means clustering, but instead of calculating a cluster mean each iteration, calculate the cluster _medioid_: the point that is the closest to all others in its cluster. 

Unlike k-means, we don't have to use Euclidean distance here. We can instead measure _angle_ between observations as a measure of dissimilarity instead. Additionally, k-medioids is less sensitive to outliers.

## Hierarchical Clustering

Hierarchical clustering is a method of clustering where a hierarchy of clusters is built. 

### Agglomerative Clustering

Also called "bottom-up" clustering. Every point starts in its own cluster, merge clusters recursively until we have a single cluster that encapsulates all points. The _linkage function_ determines the distance between two clusters, and can be defined differently. 

#### Single-Linkage Clustering

The distance between clusters is the closest distance between two points in different clusters. This is also called _nearest-neighbor clustering_.

Single-linkage tends to give unbalanced dendograms, as they are very sensitive to outliers (outliers tend not to get joined until the very end).

#### Complete-Linkage Clustering

The distance between clusters is the _furthest_ distance between two points in different clusters. This is also called _farthest-neighbor clustering_.

Using complete linkage tends to give the best-balanced dendogram; as more points go into a cluster, the representative furthest point becomes farther away. 

#### Average-Linkage Clustering

The distance between clusters is the average pairwise distance between all pairs of points in the two different clusters. This actually has a name: UPGMA, short for Unweighted Pair Group Method with Arithmetic Mean.

#### Centroid-Linkage Clustering

The distance between clusters is the (Euclidean) distance between the two clusters' centroid vectors.

## Dendogram

A dendogram is a diagram of the cluster hierarchy in which the vertical axis encodes all the linkage distances. We can cut a dendrogram into clusters by a horizontal line according to number of clusters OR intercluster distance.

### Divisive Clustering

Also called "top-down" clustering. Every point starts in a single giant cluster, and splits are performed recursively. 

## Gaussian Kernel

The Gaussian kernel is a kernel function $k(x, z) = \exp(-\frac{||x-z||^2}{2\sigma^2})$. It approximates the dot product $\Phi(x)^T\Phi(z)$, where  $\Phi(x)$ is an infinite vector: $\Phi(x) = exp(-\frac{x^2}{2\sigma^2})\begin{bmatrix}1 & \frac{x}{\sigma \sqrt{1!}} &  \frac{x^2}{\sigma \sqrt{2!}} & ... \end{bmatrix}^T$. $\sigma$ is a hyperparameter that represents Gaussian width. Larger widths mean smoother $h$ and more bias, less variance. 

A Gaussian kernelized problem computes hypothesis $h(z)$ as a linear combination of Gaussians centered at sample points. It behaves like a smoothed-out [k-NN](#k-nearest-neighbors-k-nn).

Gaussian kernels are popular because they tend to give smooth hypothesis functions.

## Normalize

Normalizing data is centering features (subtract mean), and then dividing by the standard deviation. This effectively makes each feature have 0 mean and unit variance. This is done to ensure that a feature with a very broad range of values does not dominate other features in objective functions.

It's also important to normalize if regularization is used, so that coefficients are penalized appropriately.

## Laplacian Matrix

A Laplacian matrix is a unique matrix representation of a graph. The sparsest cut of a graph can be approximated through the second smallest eigenvalue of its Laplacian by Cheeger's inequality.

Elements of $L$ are given by:

$$
L_{ij} = \begin{cases}
        \text{deg}(v_i) & i=j \\
        -1 & i \neq j \text{ and } v_i \text{ is adjacent to } v_j \\
        0 & \text{else}  
\end{cases}
$$

$L$ is symmetric and positive-semidefinite. The number of connected components in $G$ is the dimension of the nullspace of $L$: it is equal to the number of 0-eigenvalues in $L$. This number is also called the _algebraic multiplicity_ of the 0 eigenvalue.

## Low-Rank Approximation

Any matrix, utilizing its first $k$ eigenvectors and the SVD, can be given a rank-k approximation.

## Lifted Feature Vector

For a feature vector $x$, transformation $\Phi$ results in $\Phi(x)$, a lifted feature vector with more dimensions than $x$. This is most commonly applied in kernelization to get raw non-linearly-separable data into higher dimensions such that it is linearly separable in that higher dimension. Each feature of $\Phi(x)$ is just some function of features in $x$.

Many such $\Phi$ exist to make linearly separable data in higher dimensions, but not all of these functions are actually kernels.

## Principal Component Analysis (PCA)

PCA is the process of computing the principal components and using them to project the data to a different feature space. Often, PCA only keeps a few principal components (with the largest eigenvalues)

By projecting raw data onto a selective PCA basis, PCA can be used for [dimensionality reduction](#dimensionality-reduction). 

The data is centered such that it is translation invariant.

### PCA Derivations

There exist several derivations for PCA:

#### PCA Derivation 1

Fit a Gaussian to the data in $X$ (find $\mu, \Sigma$) with MLE. Then, choose $k$ Gaussian axes with greatest variance. The covariance matrix is estimated with $X^TX$, so taking the eigenvectors of that is essentially equivalent. 

#### PCA Derivation 2

Find a direction $w$ that maximizes sample variance of _projected_ data. Formally, find $w$ that maximizes the Rayleigh quotient.

#### PCA Derivation 3

Find a direction $w$ that minimizes mean _squared_ projection distance; the line that is as close to the points as possible ("closeness" being measured by _shortest_ distance from points to line, NOT vertical residual distance).

### Principal Components

For a collection of points $X \in \mathbb{R}^{n \times d}$, the principal components, or principal component directions, are $d$ _unit_ vectors, where vector $i$ is a direction that tries to maximize variance in the data's _feature values_. Here, a best-fitting line is defined as one that minimizes the average squared distance from the points to the line; it is the line that is as close to all the points as possible. These unit vectors are all mutually orthogonal.

Principal components are eigenvectors of $X$'s covariance matrix; they are eigenvectors of $X^TX$. Principal components can be computed by [eigendecomposition](#eigendecomposition) on $X^TX$ or (more commonly) [SVD](#singular-value-decomposition-svd) on $X$. Generally, we center variables in $X$ before finding principal components. 

## Singular Value Decomposition (SVD)

A factorization of a matrix $X$ as $X = U \Sigma V^T$, which always exists for $X$. Matrices $U, V$ are orthonormal matrices whose columns contain left and right singular vectors respectively. $\Sigma$ is a diagonal matrix of nonnegative [singular values](#singular-values) of $X$. We can prove via eigendecomposition that columns of $V$ are eigenvectors of $X^TX$, while columns of $U$ are eigenvectors of $XX^T$. 

The singular values of $X$, and thus $\Sigma$, are unique to $X$.  However, $U$ and $V$ are _not_ uniquely determined, so an SVD for $X$ is not unique.

It is important to note that $\text{rank}(X) = \text{rank}(\Sigma)$: there will be one non-zero eigenvalue per linearly independent column in $X$.

### Singular Values

Singular values of a matrix $X$ are the square roots of eigenvalues of $X^TX$. They are also the diagonal elements of diagonal matrix $\Sigma$ in the SVD of $X$.

## Spectral Clustering

In spectral clustering, we want to cluster a graph $G$, where the weight of edge $(i,j)$ represents the similarity between vertex $i$ and vertex $j$. We want to cluster vertices connected with high edge weights together.

The problem:

```{prf:definition} Spectral Clustering Optomization Problem
Find cut-vector $y$ that minimizes $y^TLy$ subject to constraints $y^Ty = n$ (relaxed binary constraint) and $1^Ty = 0$ (balance constraint). 
```

Note the minimization problem + relaxed binary constraint is equivalent to minimizing the Rayleigh quotient of $L$ and $y$. 

To solve this problem:

```{prf:algorithm} Spectral Clustering Algorithm
1. Compute Laplacian matrix $L$ for $G$.
2. Compute Fiedler vector $v_2$ of $L$
3. Round $v_2$ with a [sweep cut](#sweep-cut). Choose min-sparsity cut as final cut $y$. 
```

### Sweep Cut

In a sweep cut, we choose the cut from our Fiedler vector $v_2$. First, we sort components of $v_2$ in ascending order. Then try each adjacent cut between successive components, and calculate the cut weight if that cut is chosen. We choose the cut that gives the smallest cut weight. 

### Spectral Clustering (Vertex Masses)

In the case that we have vertex masses indicated by $n \times n$ diagonal matrix $M$, our constraints change: 

```{prf:definition} Spectral Clustering Optomization Problem with Masses
Find cut-vector $y$ that minimizes $y^TLy$ subject to constraints $y^TMy = \text{Mass}(G) = \sum_iM_{ii}$ and $1^TMy = 0$. 
```

The solution is the Fiedler vector of generalized eigensystem $Lv = \lambda Mv$.

## Sparsest Cut

The sparsest cut of a graph $G=(V,E)$ is a cut that partitions $G$ into two subgraphs $G_1,G_2$ that minimize the [sparsity](#sparsity) of the cut.

### Sparsity

The sparsity of a cut is the ratio of total weight of cut edges divided by the number of vertices in the smaller half of the partition. We could also assign _masses_ to vertices with diagonal mass matrix $M$. We want cuts that minimize the number of edges crossed AND be as close as possible to a bisection  .

## Support Vector Machines (SVM)

SVMs are supervised binary linear classifiers. For data points with $d$ features, SVMs find a $(d-1)$-dimensional hyperplane to separate classes. Generally, we choose the maximum-margin hyperplane: the hyperplane that has the largest separation between points in classes. Once we find our optimal weights $w, \alpha$, our classifier is given as $h(x) = \text{sign}(w^Tx + \alpha)$. 

In the case that our data is not linearly separable, we may want to map feature vectors to a higher-dimensional space, then find a linear hyperplane there. To ensure dot products between pairs of feature vectors can be computed in terms of features from the original space, SVMs are often kernelized: dot products are replaced by nonlinear kernel functions. Points $x$ in a feature space are then mapped onto a higher-dimensional space by a weighted sum of kernel computations, one for each point in the training set. 

### Hard-Margin SVM

Hard-margin SVMs should only be used for linearly separable data.

To find the MMH, hard-margin SVMs find two parallel hyperplanes, called _margins_, that maximally separate data. The MMH lies halfway in between the margins. The "right" margin is defined by $w^Tx + \alpha = 1$, and the "left" margin as $w^Tx + \alpha = -1$. We represent these as constraints: if a point $X_i$ is in class C ($y_i = 1$), then $w^TX_i + \alpha \ge 1$. If it is not ($y_i=-1$), then $w^TX_i + \alpha \le -1$. We can represent all of these as a one line constraint: $y_i(w^TX_i + \alpha) \ge 1$ for all training points $i$.

These margins will pass at least 2 points; as these points determine the margin hyperplane, they are called _support vectors_.

The distance between the margins is $\frac{2}{||w||}$. Thus, to maximize distance, we want to minimize $||w||$.

Overall, the hard-margin SVM tries to solve:

```{prf:definition} Hard-Margin SVM problem
:label: HMSVMproblem

The _hard-margin SVM problem_ for a dataset $X \in \mathbb{R}^{n \times d}$ is to find $w, \alpha$ that minimizes $||w||^2$ subject to constraint $y_i(w^TX_i + \alpha) \ge 1$ for $i \in 1,...,n$.
```

## Discriminative Models

Discriminative models attempt to model the posterior probability $P(Y|X)$ for a class directly, skipping all distribution and prior probability stuff.

## Generative Models

Generative models assume training points come from probability distributions. __These models attempt to predict these class-conditional distributions (one per class)__ so it can make predictions on future data. Given the density and prior, we can use Bayes' formula to estimate the posterior $P(Y|X)$ for each class. The class with the highest posterior (* asymmetric loss) is the official prediction.

In both LDA and QDA, we estimate conditional mean $\hat{\mu}_C$ & conditional variance $\hat{\sigma^2}_C$ for each class C.

## Neural Network 

A neural network is a machine learning algorithm that models biological neural networks in animal brains. 

### Activation Functions

The activation function in a neural network is a (nonlinear) function that is processed on the node input(s). In biological neurons, the activation function represents the rate of firing.

#### Linear

Makes the model (at least for that layer) behave as it was a linear classifier. So if you want to capture nonlinear patterns, never use this.

#### ReLU

Probably the most popular activation function for deep neural networks. Advantages:
- Sparse activation of nodes; only around half of hidden units are activated (nonzero output) at a time. 
- Avoids [vanishing gradient](#vanishing-gradient-problem) problems, compared to sigmoid, as it saturates only in the positive direction (instead of both)
- Fast computation

#### Sigmoid

Sigmoid activation functions always go hand-in-hand with posterior probability problems. So in NNs where knowing that is important, we use sigmoid. Unfortunately, it is very susceptible to the vanishing gradient problem. 

#### Softmax

Softmax is used for classification where we have $k \ge 3$ classes. For this, we need $k$ output units, each outputting a probability that the input belongs to the associated class.

Note that softmax has a normalization constant that ensures all outputs sum to 1. Thus, in the special edge case for classification problems _where a training point can have multiple labels_, we'd want to choose sigmoid over softmax.

## k-d Tree

A k-d tree is a binary tree in which every leaf node is a k-dimensional point. Every internal node implicitly splits the feature space into two half-spaces.

### Linear Discriminant Analysis (LDA)

Linear decision boundaries. Assumes every class-cond distribution is multivariate Gaussian. Assume same variance (width) $\sigma$ for each Gaussian- i.e. same covariance matrix $\Sigma$ for each class. So that means we calculate a single _pooled within-class variance_ for all classes.

We come up with the linear discriminant function by applying MLE to the log-posterior $P(Y=C|X=x) = f_C(x)\pi_C$. The decision boundary between two classes C and D consists of all points $x$ where $\Delta_C(x) = \Delta_D(x)$. 

To classify, calculate linear discriminant function for each class, then pick class w/ max value.    

### Quadratic Discriminant Analysis (QDA)

Quadratic decision boundaries. Estimate conditional mean $\hat{\mu}_C$ & a __different conditional variance $\hat{\sigma^2}_C$ for each class C__. 

The decision boundary between two classes C and D consists of all points $x$ where $\Delta_C(x) = \Delta_D(x)$, except now $\Delta_C(x)$ is the _quadratic discriminant function_. This decision boundary is now quadratic. To classify, calculate quadratic discriminant function for each class, then pick class w/ max value.

QDA's bigger flexibility with the covariance matrix can fit data better than LDA. Of course, overfit is always an issue.
## Prior Probability

The prior probability is the probability of an event at the start. This is usually calculated for each class, and is equal to what proportion of sample points belong to a certain class. We denote the prior probability of class $k$ as $P(Y=k)$, or sometimes just as $\pi_k$.

## Posterior Probability

The posterior probability is the updated probability of an event _after_ some kind of data/evidence is collected. The posterior probability of class $k$ given evidence $X$ is denoted $P(Y=k|X)$. 

Many classifiers, including Bayes classifier, LDA, and GDA, involve calculating the (weighted if asymmetric loss) posterior probability for each class given a test point $X$ and predicting the class with the highest posterior probability. In generative models, maximizing the posterior is equivalent to maximizing $f_C(x)\pi_C$ for a class C. 

## Spectral Graph Clustering (Multiple Eigenvectors)

We can use $k$ eigenvectors (solutions of $Lv = \lambda Mv$) to cluster a graph into $k$ subgraphs. We scale each eigenvector $v_i$ such that $v_i^TMv_i = 1$, so that $V^TMV = I$ (eigenvectors are columns). Row $V_i$ is called the _spectral vector_ for vertex $i$. We normalize each row to have unit length; now, each spectral vector is a point on a unit hypersphere centered around the origin. 

Then, we k-means cluster these spectral vectors, and thus the vertices they belong to. k-means clustering will cluster together vectors that are
separated by small angles, since vectors lie on the sphere. 

### Soft-Margin SVM

The soft-margin relaxes constraints such that it can still find a boundary for non-linearly-separable data. To do this, it introduces _slack terms_ $\xi_i$, which are values proportional to how much a point $X_i$ violates the decision boundary.

So now, the problem to solve changes: 

```{prf:definition} Soft-Margin SVM problem
:label: SMSVMproblem

The _soft-margin SVM problem_ for a dataset $X \in \mathbb{R}^{n \times d}$ is to find $w, \alpha, \xi_i$ that minimizes $||w||^2 + C\sum_{i=1}^{n}\xi_i$ subject to constraint $y_i(w^TX_i + \alpha) \ge 1 - \xi_i$ for $i \in 1,...,n$. An additional constraint is that all slack terms must be positive: $\xi_i \ge 0$ for all $i$. 
```

$C$ is a hyperparameter: for large $C$, we penalize slack more heavily, resulting in finer adjustments to the decision boundary and increasing variance. Infinite $C$ is effectively a hard-margin SVM (which allows no slack). For small $C$, we penalize less and we can get a wider margin $\frac{2}{||w||}$, but also allow for much more misclassification.

For soft-margin SVM boundaries, look for big-time margin bounds with the exception of (a very few) violating points. 

## Random Forest

A Random Forest is an ensembler where base learners are decision trees. Each base tree is given extra randomization during training; each node of each tree selects a subset of $m$ features remaining. Because of the ensembling and randomization of trees, random forests tend to overfit less than decision trees.

## Regularization

Regularization are techniques used for error reduction by reducing overfitting. It does so by adding a penalty term to the cost function. The optimal solution is the tangential intersection between isocontours of the regularization constraints and the (least-squares) error function.

### Ridge Regression

Ridge regression is the addition of the L2 penalty term on a weight vector $w$ to the cost function. The ridge regression estimator is $\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$, where $\lambda$ is a hyperparameter related to the amount of penalty enforced. The ridge regression estimator is unique; thus, ridge regression is a means of solving ill-posed problems.

Ridge regression shrinks weights but does not make them 0, in constrast to [LASSO](#lasso). 

Note that L2 regularization can be thought of as a constraint, with hyperspherical isocontours in feature space.

### LASSO

LASSO is the addition of the L1 penalty term on a weight vector $w$ to the cost function. Unlike ridge regression, LASSO does not have a closed-form solution. The LASSO estimator is unique when $\text{rank}(X) = p$, because the criterion is strictly convex. Unlike ridge regression, LASSO also tends to set weights to 0. Larger values of hyperparameter $\lambda$ tend to lead to more zeroed weights. Because features with weight 0 are essentially useless, LASSO is a means of feature selection.

Note that L1 regularization can be thought of as a constraint, with cross-polytope isocontours in feature space.

## Variance

For a machine learning model, variance is error that results from model's sensitivity to small fluctuations in the training set. Models with high variance lead to algorithms fitting to noise in the dataset rather than the true relationship. Models with high variance are said to overfit. 

Generally, models with high __model complexity__ have high variance. Complexity generally results from added features and finely tuned hyperparameters (model architecture).

## Vanishing Gradient Problem

The vanishing gradient problem is when NN weights are too close to 0 or 1, resulting in their loss gradients being way too small. This can even go so far as effectively halting NN training completely. 

We see this problem for early layers in deep neural networks, as gradients of these early layers are products of gradients in later layers. Since gradients are generally between $[0,1]$, since there are more terms this gradient might converge to 0.

Many solutions exist to solve this problem:
- Use ReLU over sigmoid.
- Choose smaller random initial weights for units with bigger fan-in.
- Set target values (in $y$) to $[0.15, 0.85]$ instead of $[0,1]$. This will help with a stuck output layer, NOT hidden layers.
- Modify [backpropagation](#backpropagation) to add a small constant to $s'$.
- Use cross-entropy loss instead of squared error, which has a much larger gradient to compensate for the vanishing gradient of sigmoid. This will help with a stuck output layer, NOT hidden layers. 

## Saturation (Neural Networks)

Saturation in neural networks means that hidden layer outputs have output values close to 0 or 1. This means that input values have a very high absolute value. As a result, weights will not change much during training, and it will be very slow. Most common with sigmoid activation function.

Saturation is also a sign of overfit. It is 

## Weak Classifier

A weak classifier generally refers to binary classifiers, which do little or no better than 50% accuracy on unseen data points. In other words, a human guessing randomly could do just as good. 

