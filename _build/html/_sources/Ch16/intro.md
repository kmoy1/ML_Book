Neural Networks
=========================

## Intro

The shining star of learning algorithms are neural networks. Neural networks can do both classification and regression. 

Neural networks are a culmination of the topics we've discussed in this book so far: 

- Perceptrons
- Logistic Regression
- Ensemble Learning
- Stochastic Gradient Descent
- Lifting sample points to a higher-dimensional feature space

Neural nets have an added super cool benefit of being able to **learn features on their own**. 

Let's go back a few chapters: all the way back to perceptrons. Remember that perceptrons were basically machines that came up with a linear decision boundary. Of course, there are inherent limitations to what a perceptron can do, particularly XOR:

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210418152811444.png" alt="image-20210418152811444" style="zoom:33%;" />

Here, we simply convert the XOR truth table to four points in 2-dimensional (binary) feature space. Blue points correspond to 1, while white points correspond to 0? Note that no matter how hard you try, you cannot find a linear separator that separates blue from white. 

The fact that the perceptron showed such limitations on even "simple" problems like these drastically slowed research in the neural network field, for about a decade. Tough.

But there exist many simple solutions for this. One is  adding a quadratic feature $x_1x_2$- effectively lifting our points to 3-dimensional feature space. *Now*, XOR is linearly separable.

But what about another way? Say we want to calculate $X$ XOR $Y$. We know that perceptrons output a linear combination of its inputs. What if we *_chained_* multiple perceptrons together like this, for inputs $X$ and $Y$. Can $Z$ be the XOR of $X$ and $Y$? 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210418153907912.png" alt="image-20210418153907912" style="zoom:50%;" />

No. In all, all we have from chaining linear combos in this manner is just another linear combination. So equivalently, we just have another perceptron- will only work for linearly separable points. 

We need to implement in between the initial perceptron outputs. Let us call the linear combo boxes *neurons*, although they may sometimes be referred to as _units_. If a neuron's output is run through some nonlinear function *before* it goes to the next neuron as an input, then we may be able to simulate logic gates, and from there we might be able to build XOR. 

There are many choices for this nonlinearity function. One that is often used is the logistic function. Remember the logistic function has range $[0,1]$,  which is like an inherent normalization that ensures other neurons cannot be overemphasized. The function is also smooth: it has well-defined gradients and Hessians we can use for optimization.

Here's a two-level perceptron with a logistic activation function that implements XOR: 

<img src="C:\Users\Kevin\AppData\Roaming\Typora\typora-user-images\image-20210418155127858.png" alt="image-20210418155127858" style="zoom: 50%;" />

Can an algorithm learn a function like this? 

## Neural Networks with 1 Hidden Layer

Remember with our bias term, our inputs layer has $d+1$ units. Then, we have a **hidden layer** of perceptrons- again, accounting for the bias term, we have $m+1$ units. Finally, our output layer will have $k$ units- the number that $k$ represents is application-dependent. 

Each layer's weights can be represented by a matrix with each row representing a node in the _next_ layer, and each column representing a node in the *current* layer. Therefore, for our first layer, we will have a $m \times (d + 1)$ sized matrix $V$, with each element representing a connection weight between nodes in the input layer and hidden layer. Similarly, the weight matrix connecting the hidden layer to the output layer has size $k \times (m+1)$. We will denote this matrix $W$. 

Assume our activation function is the logistic function, but many other activation functions can be used here. 

We can denote our output vector as a function of the input:

$$z = f(x) = s(Wh) = s(Ws_1(Vx))$$

where $h = s_1(Vx)$, and $s_1(x)$ is the application of the activation function to the output layer WITH bias, and $s(x)$ is the application of the activation function to output vector $x$. 

Neural networks often have more than one output. This allows us to build multiple classifiers that share hidden units. One of the interesting advantages of neural nets is that if you train multiple classifiers simultaneously, sometimes some of them come out better because they can take advantage of particularly useful hidden units that first emerged to support one of the other classifiers.

We can add more hidden layers, and for image recognition tasks it’s common to have 8 to 200 hidden layers. There are many variations you can experiment with—for instance, you can have connections that go forward more than one layer. 

## Training Neural Networks

We usually utilize stochastic or batch gradient descent to train neural networks. We need to pick a loss function $L(z, y)$: usually, this is the squared norm: $L(z,y) = ||z-y||^2$, and cost function $J(h) = \frac{1}{n}\sum_{i=1}^{n}L(h(X_i), Y_i)$. Note that a single output for a data point is not a scalar but a vector. Therefore, outputs $Y_i$ is a _row_ of output matrix $Y$. Sometimes there is just one output unit, but many neural net applications have more.

The goal is to find the optimal weights of the neural network that minimize $J$: specifically, the optimal weight matrices $V, W$ (there are more in NNs with more than one hidden layer, of course). 

For neural networks, generally there are many local minima. The cost function for neural networks are generally not even close to convex. For that reason, it’s possible to end up at a bad minimum. In a later note, we'll discuss some approaches for getting better minima out of neural nets. 

So what's the process of training? 

Let's start with a naïve approach. Suppose we start by setting all the weights to zero ($W, V = 0$), then apply gradient descent on the weights. Will this work? 

Neural networks have a symmetry: there’s really no difference between one hidden unit and any other hidden unit. So if we start at a symmetric set of weights, the gradient descent algorithm won't ever break this symmetry (unless optimal weights are symmetric, which is incredibly unlikely). 

So to avoid this problem, **to train neural networks we start with random weights.**

Now we can apply gradient descent. Let $w$ be a vector of all weights in $V$ and $W$. In batch gradient descent: 

```
w = randomWeights()
while True:
	w = w - epsilon * gradient(J(w))
```

Note that in code, you should probably operate directly on $V,W$ instead of concatenating everything for the sake of speed. 

Additionally, it’s important to make sure our initial random weights aren’t too big: if a unit’s output gets too close to zero or one, it can get “stuck,” meaning that a modest change in the input values causes barely any change in the output value. Stuck units tend to stay stuck because in that operating range, the gradient $s_0(·)$ of the logistic function is close to zero.

How do we compute $\nabla_w J(w)$? Naively, we calculate one derivative per weight, so for a network with multiple hidden layers, it takes time linear in the number of edges in the neural network to compute a derivative for one weight. For example, take the neural network below, with edges labeled: 

Multiply that by the number of weights. So we get runtime $O(\text{# edges}^2)$, and backpropagation takes $O(\text{# edges})$.  With complicated neural networks, this will get bad _very_ quickly. 

## Computing Gradients for Arithmetic Expressions

The gradient of a neural network is a vector of all the partial gradients with respect to each input: 

## Backpropagation

Backpropagation is the second step involved in training the weights of neural networks, via calculating the gradient of the *error function* with respect to the neural network's weights. It utilizes dynamic programming to calculate ALL these gradients (partial derivatives) in runtime linear to the number of weights. 

