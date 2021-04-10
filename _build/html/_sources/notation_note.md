# A Quick Note on Notation and Terminology

Throughout this book, we adhere to various standards in using variable names to represent certain numerical concepts. This section is purely meant for convenience and reference as you read the book, so feel free to skip this section if you wish. 

- $n$ represents the number of observations/data points in a dataset. 
- $d$ represents the number of features (aka "predictors") for each data point in a dataset. Each observation is thus represented as a point in $d$-dimensional space. We can call such points _feature vectors_ or _sample points_. 
- $C$ is the class for which a classifier tries to predict if a data point is in or not. Sometimes, I may use $D$ to represent the "not in $C$" class.
- Assume vectors are column vectors unless otherwise stated: $x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ ... \end{bmatrix} = \begin{bmatrix} x_1 & x_2 & x_3 & ... \end{bmatrix}^T $
- Uppercase Roman letters, like $X$, are used to denote matrices, random variables, or sets.
- Lowercase Roman letters, like $x$, denote vectors.
- Greek letters, like $\alpha$, denote scalars.
- $i, j, k$ are reserved for indices: summations, element-wise matrix definitions, etc.
- Inner products can either be written as $x \cdot y$ or $x^Ty$ for two vectors $x, y$ of the same length. 
- The Euclidean norm of a vector $x$ is denoted as $||x|| = \sqrt{x \cdot x} = \sqrt{x_1^2 + x_2^2 + ... + x_d^2}$, and represents the length of the vector.
- "Normalizing" a vector means dividing a vector by its norm, i.e. $\frac{x}{||x||}$, which makes its new norm 1.
- $H$ refers to a hyperplane, which always has $d-1$ dimensions in $d$-dimensional space. For example, it is a (1D) line in 2D space, a (2D) plane in 3D space, etc.