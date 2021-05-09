Neural Network Variations
====================================

Let's analyze a few basic variations on the standard neural network.

First, remember that neural networks can do regression as well as classification. Unlike classification, neural networks performing regression generally do _not_ have nonlinear activation functions. Do note that in this case the backpropagation algorithm will change as well- fortunately, in a way that makes it simpler. 

For classification, if we have two classes, the sigmoid activation function is perfectly fine. What about more than 2 classes, like MNIST? In this case, the __softmax function__ might be a better alternative. The softmax function is different in that it is a function of $k$ outputs, not just one. So now, with $k$ unique classes, we want $k$ units in the output layer. Let $t = Wh$ = the length-$k$ output units vector before softmax. We then take $t$ and run it through $k$ softmax output $z_j(t) = \frac{e^{t_j}}{\sum_{i=1}^{k}e^{t_i}}$. Again, there will be $k$ softmax outputs, each which gives a number in range $z_j \in (0,1)$. Of course, this is designed such that the sum of all softmax outputs is 1. We can think of the softmax output, then, as a posterior probability for each class. 

Note that the softmax function applied on two classes is essentially equivalent to sigmoid.

Let's derive backpropagation with softmax outputs. First, we compute the derivative of the softmax output $z_j$ with respect to its input $t_j$: 

$$
\frac{\partial z_j}{\partial t_j} = z_j(1-z_j)
$$

We also need the derivative of the softmax output with respect to softmax inputs that _do not_ come from it: 

$$
\frac{\partial z_j}{\partial t_i} = -z_jz_i
$$

for $i \neq j$.

These are the derivatives we need to do backpropagation! But before that, let's talk about one of the issues with sigmoid outputs.

## Sigmoid Unit Saturation

One issue that comes from using sigmoid is that they can __saturate__. While it does do a good job with keeping other neurons from being able to "overpower" other neurons downstream, the issue is that the output of a unit can get "stuck" if it is too close to 0 or 1. 

When unit output $s$ is close to 0 or 1, that means $s' = s(1-s) \approx 0$. This means gradient descent will be extremely slow for that particular neuron. This unit is "stuck", and can slow down training by quite a bit. Additionally, we can also output a bad local minima!

With more hidden layers, the risk of saturation gets larger. That's why you'll find it very rare for image recognition networks, which can have hundreds of hidden layers, to have sigmoid outputs. 

```{note}
This is commonly referred to as the __vanishing gradient problem__.
```

There are a few ways to mitigate the vanishing gradient problem: 

1. Initialize weights based on the _fan-in_ of units they connect to. Let's say we have a NN unit with _fan-in_ $\eta$: it has $\eta$ input edges. We can initialize each incoming edge to a random weight with mean 0, standard deviation $\frac{1}{\sqrt{\eta}}$. The larger the fan-in, the easier it is to oversaturate a unit. So for larger fan-ins we need a smaller initial weight. 
2. Change target values from binary (0/1) to 0.85 and 0.15. If our target is 0/1, then output units will of course tend towards 0/1, and thus be saturated. The region $[0.15, 0.85]$ is the non-saturated region for sigmoid outputs.
3. In the case that _hidden_ units are saturated, we can modify backpropagation to add a small constant (typically around 0.1) to $s'$. This helps the gradient not be 0. We find that we often find a better descent direction with this than steepest descent!
4. Use cross-entropy loss instead of squared error. Now, the gradient actually goes to _infinity_ as predictions $z_j$ get close to 0 or 1. It is strongly recommended that for any given sample point, its $k$ labels add up to 1: $\sum_{i=1}^{k}y_i = 1$. For example, in MNIST, if we have an input image 9, we assign $y_9 = 1$ and everything else to 0. You can think of these as posterior probabilities.

There also exists a cross-entropy loss for sigmoid outputs. For a single sigmoid output $z$, we have $L(z,y) = -y\ln z - (1-y)\ln(1-z)$.

How can we do backprop for a $k$ softmax outputs? Well, we need to compute some derivatives. Try these yourself:

$$
\frac{\partial L}{\partial z_j} = -frac{y_j}{z_j} \\
\nabla_{W_i} L = (z_i - y_i)h \\
\nabla_{W_i} L = (z-y)h^T \\
\nabla_hL = W^T(z-y)
$$

where $W_i$ is row $i$ of weight matrix $W$, and $h$ is our vector of hidden unit outputs. Remember it is _absolutely_ important that $\sum_{i=1}^{k}y_i = 1$ for these derivations to hold!!

Note the formulas for $\nabla_{W_i} L$ and $\nabla_hL$ are true for both softmax and sigmoid outputs.

Now let's derive backpropagation for a neural network with softmax outputs, cross-entropy loss, and L2 regularization.

<!-- TODO: Run through backprop algorithm, starting at 1:07:00 -->