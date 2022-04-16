# Lecture 15: Decision Trees Cont.

## Decision Tree Variations

There's almost limitless ways to vary a decision tree. 

First, we can utilize **multivariate splits**. In normal decision trees, each node splits on ONE feature. However, we can split on multiple if necessary. In another way, we find *non-axis-aligned splits* (where axes are features in feature space), instead we choose some other split. We can utilize other classification algorithms like SVMs or GDA to find such decision boundaries. We could even generate a bunch of random splits and pick the best one.

First, let's visualize a standard decision tree which splits only on one feature at each node:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326164607386.png" alt="image-20210326164607386" style="zoom: 50%;" />

Of course, a better split exists:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326164727718.png" alt="image-20210326164727718" style="zoom:50%;" />

Note that this tree only has TWO child nodes, but now we're considering two features. 

There is a tradeoff, though, to multivariate splits. First, you might gain better generalization of test data, but there might be worse interpretability (than splitting one feature at a time). You might also lose a lot of speed, especially with a very large feature space. Think if you had 10000 features: then, if you use an SVM-created split, a single node must look at ALL 10000 features, and this will of course add up over all nodes and test points. 

So what we can do is compromise: we can just limit the number of features we split on. How can we choose this limit? Forward stepwise selection, or even Lasso, can work to find optimal subset of features. 

## Decision Trees in Regression

How can we do regression with decision trees? We can use a **decision tree to define a piecewise constant regression function.**

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326165401252.png" alt="image-20210326165401252" style="zoom:50%;" />

Above is the function we might get if we apply piecewise constant regression with some training data. We assign "boxes" of our piecewise function as the average of the sample points they contain. 

We also need rules for creating splits in our decision tree. Recall a cost function (entropy) and chose the split that minimized the weighted average cost. We can do the same thing with regression, but with a slightly different cost function. Our entropy function now needs to be extended to quantitative features and continuous data. So now our cost function $J$ will just be mean squared error. Given $S$, we calculate $J(S) = \frac{1}{|S|}\sum_{i \in S}(y_i - \mu_s)^2$ where $\mu_s$ is the mean for sample points in $S$. So we look at all possible splits ($i$) and use the split that minimizes $J(S)$. 

## Pruning

Another important thing to think about is **stopping early**, or **pruning**. Last lecture, we went over the basic top-down tree construction algorithm. However, there might be a time where we want to STOP construction of nodes early: i.e. we might stop before pure leaf nodes are made. There are many reasons to do this: 

- Limiting tree depth, for speed
- Limit tree size for speed (esp. with a ton of sample points)
- Complete tree might overfit data a lot.  
- Given noise, with fewer leaves outliers are given greater emphasis to sway classification. 
- Overlapping distributions: for example in GDA, we had class C's gaussian overlap with class D's gaussian. In regions of overlap, you might just want a majority vote instead of further nodes. Specifically, it might just be better to estimate posterior probabilities with non-pure leaves. 

Below is a visualization of the last point: 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210329131235413.png" alt="image-20210329131235413" style="zoom:50%;" />

On the left, the cuts represent our decision boundaries, and also indicate our tree has exactly three leaf nodes. Let's look at the rightmost leaf node (with green points). Notice the overlap: a few red points and a blue point along with all the green. On the right is a histogram showing these counts, which indicate posterior probabilities. Of course, we want our leaves to have an adequate number of sample points for this. 

So those are the reasons to prune. Let's now focus on an algorithm to do so. We have to ask: what specific *stop conditions* do we need to apply in tree building? Specifically, when do we stop splitting? We can decide to not split if that split doesn't decrease cost- i.e. doesn't reduce (weighted) entropy or error. This is dangerous, though- there are situations where using this rule can cause you to stop too early. Another stopping condition we might try: we stop when *most of*  the sample points in a node have the same class- say, above 95%. Outliers in these nodes are just considered anomalies and ignored. Another stopping condition: stop when a node has very few sample points- say, less than 10. Another way is to look at the size of the cell's edges. Another way is to limit depth of the tree itself, but this is risky if you wind up with a super-mixed node at that depth and thus dubious classification. Finally, we can just validation: we can just try out different splits and see if it improves validation error. Of course, validation is the slowest since we're building new trees each time, but there are ways to make it not-that-slow. 

Again, the reason we prune is because we are worried about overfitting. Validation is a *great* way to determine the degree of overfit. But the grow-too-large, then prune strategy is great here to combat overfitting. 

In the case of classification, we have two options. In the case of classification, we have two options. We can return a majority vote, OR we can return the class with the highest posterior probability. If we are doing regression, we just return the average of all points in that leaf node. 

## Pruning, Cont. 

Pruning is used when overfit. Instead of stopping early, though, we overgrow the tree but then greedily prune it back, removing some of the splits we made IF the removal improves validation performance (or keeps it the same- smaller tree is generally better). This is much easier and more reliable than trying to guess in advance whether to split or not. 

We are constantly checking leaf nodes and removing BOTH the leaf node and sibling node if validation isn't improved by having that split. We cannot prune nodes that have children, so we need to go bottom-up. 

Isn't validation expensive though? It is, but there is a shortcut. For each leaf node, make a list of all training points included in that node. This way, we don't have to look at *all* validation points at pruning: we only have to look at the points included by the sibling nodes. If pruning is decided, the two lists would merge into the parent node. We must ask ourselves how validation accuracy would change after this pruning. Thus it is quite fast to decide whether/not to prune a pair of sibling leaf nodes. 

It is quite common that a split that didn't make much progress is followed by a split that *does* make a lot of progress, because the first split *enabled* the second big-info split to happen. 

Let's look at an example: predicting the salaries of baseball players.

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210329134811672.png" alt="image-20210329134811672" style="zoom:50%;" />

The two features that had the most predictive power of salary was longevity and hitting average of each baseball player. Note that on the left graph, "tree size" actually refers to the number of tree leaf nodes, which is representative of tree complexity. Naturally, training accuracy gets better as we refine the tree, but validation accuracy isn't as good. The graph seems to show that 3 leaf nodes was optimal. 

## Ensemble Learning

Decision trees have a lot of pros: fast, conceptually simple and human-interpretable, invariant under translation and scaling, and **quite robust to irrelevant features:** if we have useless features, the decision tree just ignores them (doesn't split on them).

However, the tree's biggest weakness is that they just aren't the best at prediction. This is because **decision trees tend to have high variance** (but can have very low bias). For example, suppose you have a training set which you split into half, then train a decision tree on each half. It is not uncommon for these two trees to be *very* different. How can we fix this? 

Let's take a crude analogy. Suppose we are generating random numbers from a distribution. ONE sample might have very high variance. But generating $n$ random numbers all from the same distribution, then we have a much better estimate for the mean of the distribution itself. We can do the same thing with classifiers- more classifications from different sources could be much better for accuracy!

We call a learning algorithm a *weak learner* if it does only just better than guessing randomly. Combining, or *ensembling*,  many weak learners can result in a strong learner. There are many ways we can do ensembling. We can take the average of the learners' predictions. We can use completely different learning algorithms as the base learners. But this might take a lot of code/runtime, so another idea is to use the same learning algorithm, just trained on different training sets. This method can work well IF we have a lot of data to work with. 

What about if we *don't* have that much data? The first idea is **bagging**: same learning algorithm on many *random subsamples* of the single training dataset. Another idea is called **random forest**: it includes bagging with many different decision trees, but also force each decision tree to be further randomized (parameters) during construction.

Once we have the output, how can we take the average? For regression, can either take the mean/median of all outputs- which is decided by validation. For classification, we can take a majority vote, OR if the base learners output posterior probabilities, we can average those and classify based on that. 

First, we want to try and use learners that have low bias. For example, deep decision trees. It is okay if these base learners have high variance- ensembling inherently reduces variance by averaging! Each base learner overfits in its own unique way- but this disappears upon averaging.

You cannot count on averaging reducing bias. Sometimes it does, sometimes it doesn't. **Averaging a bunch of linear classifiers results in a nonlinear decision boundary.** 

Another thing to think about: hyperparameter settings for base learners are generally different from the ensemble. This is because the hyperparameters have a big influence on bias-variance tradeoff. So generally, we tune hyperparameters to lower bias (and higher variance). Note the number of trees itself is a hyperparameter. 

## Bagging

Bagging is actually short for Bootstrap Aggregating. We create many base learners, same algorithm, with a single training set. Most commonly used with decision trees as base learners, but can be others too. 

The idea of bagging: given a training set of length $n$, generate a bunch of random subsamples with size $n'$ (usually equal to $n$), **sampling with replacement**. This means we can have duplicates of points in our subsamples! The purpose of this is to increase variety along our base learners, so we don't have identical ones. Points chosen multiple times will be given extra weight- specifically, j times as much weight as a point chosen once. Of course, some points won't be chosen at all.

Specifically, for decision trees, a point sampled $j$ times will be given $j$ times its weight in entropy. For SVMs, a point sampled $j$ times incurs $j$ times as big a penalty in the SVM objective function (if it violates the margin). For regression, if a point got chosen $j$ times, then it incurs $j$ times as much loss.

So we train $T$ learners to form a **metalearner**: the learner that combines the base learner predictions on a test point in some way- average or majority, depending on the purpose.

## Random Forests

Random forests are bagging with decision trees, and the extra randomization of these decision trees. The idea: random sampling with replacement isn't random enough for our purposes. We want to reduce the chance further that no two base learners overfit in the same way and thus resemble each other. Sometimes, we just have predictors that are just really strong- this results in the same features split at the top of every tree. For example, in 1990, more than half of all spam emails concerned Viagra. Thus even with random sampling, our dataset will still have way too much Viagra, so that'll be the split feature at the top node of each tree. So averaging doesn't reduce variance much.

So what we want to do is *force* more variation into trees so that when we take the average we end up with less variance. At *each* tree node, we take a **random sample of $m$ of our $d$ features.**  Then only consider those $m$ features as a possible split feature. We must do a different feature sample at EACH tree node. $m \approx \sqrt{d}$ works well for classification. This will give us definite variety for trees. For regression, however, we probably want more features to consider, so $m \approx \frac{d}{3}$ might be better. Of course, $m$ is a hyperparameter, so we find the best one with validation- but the given ones are good starting points. When $m$ is smaller, we impose more randomness in trees, resulting in less correlated (less similar) base trees. However, smaller $m$ means more bias; we can counter this by making the trees deeper. 

Note that through using this random subset selection, random forests inherently implement dimensionality reduction, as some features will probably end up not being used. 

How many decision trees to use? Again, another hyperparameter. Generally, though, as number of trees increase, test error generally tends to go down asymptotically. However, more trees means more expensiveness- will take much longer. Also, it really loses its interpretability.

```{note}
A variation on random forests is to generate $m$ random multivariate splits. This means $m$ different oblique lines, or even quadratic boundaries. Then they look at, like, 50 of them, and choose the best split among them. We basically take the average of a bunch of different conic section boundaries. 
```