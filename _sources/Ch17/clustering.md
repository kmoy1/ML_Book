# Clustering

In __clustering__, we want to partition data into groups, or _clusters_, such that points in a cluster are more similar to each other than points in other clusters. 

Clustering has _many_ practical applications:
- Discovery: Netflix + Spotify recommendations
- Hierarchy: Given a set of genes from various species, find a good taxonomy
- Quantization: Data compression through reducing choices
- Graph Partitioning: image segmentation, finding groups in social networks

There exist many algorithms for clustering a dataset. Let's run through some. 

## K-Means
Let's begin with the k-means algorithm Given $n$ points, the big idea is to partition those points into $k$ distinct clusters. Formally, we are assign each datapoint $X_i$ a cluster label $y_i \in [1,k]$.

For each iteration, we want to calculate the _cluster mean_ $\mu_i$ as simply the mean of all points in that cluster.

Now here's the objective function. We want to find cluster label assignments $y$ such that 

$$
\sum_{i=1}^{k}\sum_{y_j = i}||X_j - \mu_i||^2
$$

In other words, we want to minimize the sum of squared distances from points to their cluster means. 

This is an NP-hard problem. We can simply try every partition: in this case, we can solve in $O(nk^n)$ time. But this is way too slow. So instead, we try a heuristic called __k-means__. In k-means, we are doing _alternating minimization_: minimizing our objective function by changing means $\mu_i$ AND changing the $y$ at each iteration.

At each iteration: 

1. $y_j$'s are fixed. We update $\mu_i$.
2. $mu_i$'s are fixed. We update $y_j$.
3. Repeat until $y$ no longer changes after each iteration. 

Given a fixed assignment $y$, we can use calculus to show the optimal $\mu_i$ is simply $\mu_i = \frac{1}{n_i}\sum_{y_j = i}X_j$: the average of all points in cluster $i$. On the other hand, given a fixed set of cluster means $\mu$, we simply assign each sample point to the (new) nearest cluster mean. In the case where a point has equal distance to multiple cluster means, the default rule is to stay in the current cluster it's already in (or pick arbitrarily if new).

Unfortunately, this algorithm does _not_ guarantee a convergence to the optimal $(\mu, y)$. 

Note that both steps 1, 2 will _always decrease_ the objective function, unless nothing changes (then we halt). This means that our algorithm will never revert to a previous state assignment. Since there are only a finite number of clustering assignments, we know the k-means algorithm _must terminate_. The k-means algorithm is usually very fast in practice, but also finds a local, not global, minimum. 

How does the k-means algorithm initialize, though? There are multiple ways:
1. Forgy method: choose $k$ random sample points as starting $\mu_i$'s
2. Random Partition: assign each sample point a random cluster label
3. k-means++: Some other advanced shit

The outcome of k-means is actually heavily dependent on the initialization. So to get the best results, it is best to run k-means multiple times with different random starts. 

It turns out that minimizing the objective function is equivalent to finding $y$ that minimizes the __within-cluster variation__: 

$$
\sum_{i=1}^{k}\frac{1}{n_i}\sum_{y_j=i}\sum_{y_m=i}||X_i - X_m||^2  
$$

So for each cluster, we calculate the total pairwise distance between points in that cluster and sum them all up, then divide by the number of points in that cluster. We sum this total for each cluster.

The important part about within-cluster variance is that now the objective function's inputs are simply data points and their cluster label assignments.

Do we normalize the data before applying k-means? Like PCA, it is dependent on whether features have different units of measurement. 

## K-Medioids

What if we have points where we _don't_ want to use the Euclidean norm (Euclidean distance) and means?

Another alternative algorithm for clustering is the __k-medioids algorithm__. This will allow us to use other distance metrics $d(x,y)$- called the _dissimilarity function_. There are a lot of options for what you can choose for $d$, but ideally, they satisfy the _triangle inequality_ $d(x,y) \le d(x,z) + d(z,y)$. 

When would we not want Euclidean distance as a measure of dissimilarity? Suppose you run a store, and have a database that tells you how many of each product each customer bought. You want to cluster together customers who buy similar products and do market analysis on these clusters. However, Euclidean distance won't be good here: we will get a massive cluster of customers who only bought one thing. Instead, it makes more sense to _treat each customer as a vector_, and use the _angle between customers_ as a dissimilarity function. If this angle is large, there is a large dissimilarity. 

So how does this change the algorithm? Instead of calculating cluster means, we are now calculating cluster _medians_, which will always be one of the data points in that cluster. And now instead of choosing a cluster for a point based on the closest cluster mean, we now choose based on the closest cluster median.

## Hierarchical Clustering

One issue with k-means and k-medioids is we must choose $k$ _beforehand_, and there isn't really a reliable mathematical way to do this other than eyeball it. This next method, __hierarchical clustering__, is one of many clustering algorithms that tries to do this for you.

Essentially, in hierarchical clustering, what we do is create a tree of all datapoints, where every subtree represents a cluster. So some clusters will contain smaller clusters. 

There are two ways of doing this: 
- __Agglomerative clustering__: bottom-up clustering. We start with each point in a cluster, and at each iteration fuse 2 clusters into one.
- __Divisive clustering__: top-down clustering. We start with all points in a giant cluster, and repeatedly split at each iteration.

```{note}
When the input is a point set, agglomerative clustering is more. But when the input is a graph, divisive clustering is more common.
```

So now, instead of measuring distance between two points, we are measuring distance between two _clusters_. There are four options for this:
- Complete Linkage: $d(A, B) = \max\{d(w, x) : w \in A, x \in B\}$
- Single Linkage: $d(A, B) = \min\{d(w, x) : w \in A, x \in B\}$
- Average Linkage: $d(A, B) = \frac{1}{|A||B|}\sum_{w \in A}\sum_{x \in b}d(w,x)$
- Centroid Linkage: $d(A,B) = d(\mu_A, \mu_B)$

```{note}
Note complete, single, and average linkage will work for any pointwise distance function $d$, while centroid linkage really only makes sense for Euclidean distance. There does exist a variation of the centroid linkage that uses the medoids instead of the means, and
medoids are defined for any distance function. Moreover, medoids are more robust to outliers than means.
```

So in _greedy agglomerative clustering_, we are just repeatedly fusing the two clusters that minimize $d(A,B)$. Naively, we would search through every cluster pair at each iteration, which would take $O(n^3)$ time.

A __dendogram__ is a diagram of the tree in a cluster hierarchy in which the vertical axis encodes all linkage distances $d(A,B)$. Take a look at the example dendogram below:

```{image} pictures/dendogram.png
:width: 600px
:align: center
```
You can see by making a horizontal line, we can split our dendogram into an arbitrary number of clusters. The lower the line, the more (finely-cut) clusters we get. 

In most applications, we probably want to use average or complete linkage. Some things to note: 
- Single linkage is very sensitive to outliers, and tends to give a very unbalanced tree
- Complete linkage tends to be the best-balanced: as a cluster gets large, the furthest point in the cluster is already far away. So large clusters are more resistant to grow more than smaller ones.

Note that all these clustering algorithms we've gone over are _unstable_: deleting a few input points can sometimes give very di↵erent results. Despite this, these are still very commonly used in practice today. And it’s not clear to me whether a truly stable clustering algorithm is even possible.