# Decision Trees

Let's talk about classification in the form of **decision trees**.

It is more similar to k-NN classifiers than other ML methods we've learned about. The basic idea: we carve up space into rectangles. Decision trees are **nonlinear**: their decision boundaries can be very nonlinear. Decision trees are mostly used for classification, but can be used for regression too.

Decision trees, like regular trees, have two types of nodes: *internal nodes*, which represent decision points as you walk down the tree, and *leaf nodes*, which specify reaching a certain final classification/hypothesis for a given data point . Decisions at internal nodes are determined by looking at some subset of features (usually one).

Let's say we have a decision tree to decide whether or not to go for a picnic today. First, we look at an outlook feature $x_1$ that tells us weather: sunny, overcast, or raining. Based on that feature, we jump to the branch that represents sunny/overcast/raining. If overcast, predict YES. If sunny, check humidity as a feature $x_2$. If it's greater than 75% humidity, then predict NO. If less, predict YES. Now let's look at the raining subtree. If it's raining, the next thing to do is check a wind feature $x_3$. If wind is > 20 mph, then predict NO (too windy). If $\leq 20$, predict YES.

Here's this visually:
<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210325001626685.png" alt="image-20210325001626685" style="zoom:33%;" />

We can also look at the tree in terms of its features. We have three features: $x_1, x_2, x_3$. We can these features and plot areas where YES/NO is predicted. For example, the feature split plot for $x_1, x_2$ (outlook as $x_1$, humidity as $x_2$) is this, where each "block" represents a hypothesis/leaf node. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210325001821624.png" alt="image-20210325001821624" style="zoom:33%;" />

We take our *feature space* (x-space) and cutting it into rectangular cells, each representing a leaf node. This works well with *categorical features* (sunny/overcast/rainy) as well as *quantitative features.*

Another reason why decision trees are popular is that they give an interpretable result. They're good for inference: often decision trees present an idea of *why* you're making the decisions you're making. 

The decision boundary for decision trees can be very complicated. For example, look at the figure below of various decision boundaries:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326132059605.png" alt="image-20210326132059605" style="zoom:50%;" />

Imagine we have a huge number of training points, and think of green as points in class (1) and yellow as points not in class (-1). The training points in the top two diagrams are ones that have a natural linear boundary, but are *not* (feature) axis-aligned. The bottom two diagrams show boundaries that are not very separating but are axis-aligned. There are many real-world problems where the right decision boundary does tend to align with feature axes. 

The two diagrams in the left half are actually SVMs: works really well with linearly separable data, but not when it isn't. The two diagrams in the right half represent decision tree bounds: these are lines representing *feature* decision boundaries. Note that even though the boundary isn't great in the top right diagram, we can just fine-tune the tree boundaries until we don't get overlap. 

## Decision Tree Algorithm

How can we build a decision tree for classification?

First of all, decision trees easily extend from binary classification to multiclass, so there isn't much simplicity gained by only having 2 classes. But we'll assume binary classification for now. 

We use a *heuristic* to find a decision tree that fits to the input data. Finding the ideal decision tree is probably NP-hard. This is a top-down algorithm to build the tree one node at a time. First, at the top node, we'll start with a set $S$ that contains the indices of the $n$ input sample points. As we go down the tree, the child nodes will have *subsets* of the nodes in the parent. Here's the pseudocode:

```
GrowTree(S):
	if (y_i = c for all i in S and some class C):
		return new leaf(C) #called a PURE leaf- only one class. 
	else:
		choose best SPLIT FEATURE j AND SPLIT VALUE beta to split S into subsets based on j < beta, j >= beta
		create left, right child S_l and S_r based on that split
		return new node(j, beta, GrowTree(S_l), GrowTree(S_r))
```

Now that our algorithm is set, we reach the most important point: how do we choose the best split? Well, just try each possible split: try each feature, and every possible split point in that feature. We only try splits between successive data points (midpoint), so we'll have only (at most) $n-1$ data points. 

How do we evaluate these splits? Just like regression and all other ML methods, we utilize a cost function. For our set $S$, we assign cost $J(S)$. Once we've decided on a cost function $J$, we want the split that minimizes some cost $J(S_l) + J(S_r)$. Alternatively, we could compute a set-size-weighted average:

$$\frac{|S_l|J(S_l) + |S_r|J(S_r)}{|S|}$$

What is a good cost function $J$? Let's look at a couple ideas. First, a bad one: take class $C$ that has a majority of points in $S$ and split on that. Then our cost function is just the number of points NOT in class $C$. Let's say $S$ has 20 sample points in class $C$, 10 in class $D$. This means our cost $J(S) = 10$. For all possible splits, the total cost as summed in the child node will remain the same: $J(S_l) + J(S_r) = J(S)$. This is an issue: all splits are equally bad with this perspective. We definitely don't want to have similar ratios of class points in our child nodes: we want our ratios to be getting closer and closer to pure nodes as quickly as possible! 

Here's a better cost function. This is one is based on information theory: how much we're *increasing information at each split*. The idea is to measure the **entropy** of a set $S$. Let's consider a random process that generates classes, and let $Y$ be a random variable that takes on a class. Let $P(Y=C) = p_C$. The **surprise** of $Y$ being class $C$ (how surprised we are that $Y$ is $C$) is defined as $$-\log_2p_c$$. For example, the surprise of an event with probability 1 is 0- this makes sense as it shouldn't be any surprise of a certain event happening. However, an event with probability 0 has infinite surprise- which also should (sorta) make sense.

When the surprise is equal to the *expected* number of bits of information, we must transmit which events happened, assuming we transmit to a recipient who knows what the events are. For example, for an event with 50% probability, we must transmit 1 bit: 0 for happened, 1 for didn't. Note that low probability events can be transmitted with *fewer* than 1 bit- this doesn't really make sense for single events, but more for results from the expectation of *many* events.

The **entropy** of $S$ is the **average surprise** of $S$. Mathematically, we denote the entropy of $S$ as $H(S)$:

$$H(S) = -\sum_{c}p_clog_2p_c$$

where $p_c$ is the *proportion of points in S that belong to class c*. 

Let's say all points in $S$ belong to class $C$. Then, $p_C = 1$, so entropy $H(S) = 0$. We can also think of entropy as a measure of *disorder*: since everything is 100% predictable, there's no disorder here. What about half class C, half class D? In this case, $p_C = p_D = 0.5$, so $H(S) = -0.5\log_20.5 - 0.5\log_20.5 = 1$. This is maximal entropy for 2 classes. 

What about more than 2 classes? Specifically, what if we have $n$ points, all different classes? Then our entropy is simply $H(S) = -\log_2\frac{1}{n} = \log_2n$. So notice the entropy is the **number of bits to encode which class is predicted for a data point!**

Let's look at a plot of $H(S)$ for two classes. The x axis is $p_C$, y axis is $H(p)$- the entropy. We notice that our cost function is **strictly smooth and concave**. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326152739360.png" alt="image-20210326152739360" style="zoom:50%;" />

We prefer to use the **weighted average of the entropy** rather than just the sum of left and right entropies. We'll denote this as 

$$H_{\text{after}} = \frac{|S_l|H(S_l) + |S_r|H(S_r)}{|S|}$$

We want to choose the split that minimizes $H_{\text{after}}$. Alternatively, we want to *maximize our information gain* $H(S) - H_{\text{after}}$. Of course, $H(S)$ is constant, so maximizing $-H_{\text{after}}$ is of course minimizing $H_{\text{after}}$. Information gain gives you a measure of how much information your decision tree gives you about a split. It can never be negative- it can be 0, indicating a split gives you no information of getting a particular class.

Let's take one type of split into account. For $S$ with 20 in class C and 10 in class D, we have $H(S) \approx 0.918$. Let's say our split gives us a left node with 10 class C and 9 class D, and a right node with 10 C and 1 D. The left node is low-info (similar ratio) but our right node is high-info. The entropy of the left node is $H(S_l) = 0.998$: almost 1, which is pretty bad. However, $H(S_r) \approx 0.439$. Their weighted average $H_{\text{after}} = 0.793$, and we've gained some information. Specifically, our info gain is $0.918-0.793 = 0.125$. 

Now let's do another split where the ratios in the child nodes are equal: 20 C and 10 D become 10 C 5 D, 10 C 5 D. Now, our info gain is 0: this split didn't accomplish anything since the ratios are the same and no class is now more likely. This is generally the exception: we will have info gain of 0 when:

- One child node is empty with no data points (trivial, can just ignore).
- Ratio of classes is exactly the same in child nodes (as they do in parent!).

Compared to percentage misclassified in each child node, why does info gain work better? Again, our graph is strictly concave. The **weighted average** $H_{\text{after}}$ **is always on a line connecting $H(S_l)$ and $H(S_r)$**. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326154502914.png" alt="image-20210326154502914" style="zoom:50%;" />

Important note: information gain is always positive as long as $p_c$ for the *parent node* is strictly between the $p_c$ for the left child and $p_c$ for the right child.

Now let's look at the cost function where we just add percentages of misclassified points. 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210326154724706.png" alt="image-20210326154724706" style="zoom:33%;" />

Now we see that $J(\text{parent}) = J_{\text{after}}$, no matter what split we choose, and there's never any progress made.

Let's think about how we choose a split. If we have binary features $x_i$, we'll have one child where $x_i = 0$ and $x_i = 1$. What about 3+ values though? Sometimes we want a 3-way split. Sometimes we want a sequence of binary splits. Choice is application-dependent.

What if $x_i$ is quantitative? We want to sort $x_i$ values in $S$ on feature $x_i$. Then, we take a *set* of the feature values so we don't have any repeats, and only unique values of $x_i$. Let's consider sorting $S$ as linear time (radix sort or whatever). Now we try all midpoints as splits. **As we scan through this sorted list from left to right, we can update entropy in $O(1)$ time per point.** The general idea is this: at the very first split, we calculate total number of class C to the left/right, as well as total number of class D to the left/right. This takes $O(n)$ time. With this knowledge stored, though, calculating entropies at further split points takes $O(1)$ time. This is because counts as we move left/right only change by a max of 1. 

What are the algorithms for building and searching our decision tree (classifying a test point)? Let's start with classification of our test point. This is the same exact thing as (binary) tree search. We walk down our tree until we hit a leaf: the rectangular block of feature space that our test point belongs to. The worst-case time for doing this is (usually) the depth of the tree, since level checking takes constant time. Note that if we have purely binary features then our depth cannot exceed the number of features. However, if we have many numerical features where we split on one feature many times, our depth can get bad. In practice, tree depths generally are upper bounded by $O(n)$. 

Next, how long does it take to train/build a decision tree? Let's start with binary features. We know there's only $d$ (number of features) splits we can try. So we try only $O(d)$ splits at each node, then choose the best one. For quantitative features, we need to try $O(n'd)$ splits, where $n'$ is the number of points in that *particular* node. Interestingly, even though we need to try a lot more splits, **runtime is the same as for binary features.** The amount of time will always be $O(n'd)$ at a node whether its a binary or quantitative feature, since we use the entropy scan trick discussed above.

Each sample point only participates in $O(\text{tree depth})$ nodes. Think about it: if each node's time cost is $O(n'd)$, and the cost that a sample point brings to any node it participates in is $O(d)$ time. Putting it all together, the running time to build our tree is 

$$O(nd*\text{tree depth})$$

This is surprisingly good. A way to think about this: $nd$ is our design matrix "size", so it takes $O(nd)$ time to read that. Depth is usually $\log n$, so basically we have input size * log(input size), or just $n\log n$. 