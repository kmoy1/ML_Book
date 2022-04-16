# Maximum Margin Classfier

Optomization algorithms have improved a lot over the years. For example, you can get rid of deciding the arbitrary step size in gradient descent by using a modern line search algorithm: this implements a more dynamic way to take a step at each iteration instead of just a constant coefficient. Even better, we can find a better decision boundary via __quadratic programming__. We'll see what this exactly means in our discussion of the maximum margin classifier.

The __maximum margin classifier__ refers to the linear classifier that has a maximum-sized margin to all its data points. The __margin__ of a linear classifier is the distance from the decision boundary to the nearest training point. So now, we have a further constraint: not only do we want a line that linearly separates our points, we want the line that puts the maximum distance between itself and these points!

Here's an example of the maximum-margin decision boundary for a sample set of 2D points:

<!-- TODO: Put example of maximum-margin decision boundary. -->

The points closest to the boundary are called __support vectors__, and can be drawn as points on two parallel dotted lines on either side of the decision boundary, equally far away, as shown above. The distance between the boundary and one of the dotted lines is the __margin__, and is the thing we want to maximize. 

Of course, we know our decision boundary is the set of points such that $w \cdot x + \alpha = 0$. The two dashed lines on the left and right respectively are denoted by $w \cdot x + \alpha = -1$ and $w \cdot x + \alpha = 1$.

Now the constraints we enforce: $y_i(w \cdot X_i + \alpha) \ge 1$ for each sample point. 