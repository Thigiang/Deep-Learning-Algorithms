# ML/AI Algorithms

This is a self project that aims to write ML/AI algorithms from scratch while learning a deep learning course from Coursera with Professor Andrew Ng.
## 1. Logistic Regression
Logistic Regression is a binary classification. Given an input $x$ (an image), its algorithm will predict the output by calculating $\hat{y} =P(y=1|x)$.
In Logistic Regression, we use a sigmoid function $\hat{y} = \sigma(z) = \frac{1}{1+e^-z}$ where $z = w^T x +b$ where:  
      - $x \epsilon R^{n_x}$  
      - $w$ (weight) and $b$ (bias) are parameters of the algorithm. $w \epsilon R^{n_x}$ and $b$ is a real number  
In order to train the parameters w and b, we use a cost function:  
    For each training example, we define a loss function:  
        $L(\hat{y}, y) = -(ylog(\hat{y}) + (1-y)log(1-\hat{y}))$  
    Then, the cost function is the average of all the loss functions on the entire training set:  
        $C=\frac{1}{m} L(\hat{y}, y) = -\frac{1}{m} (\sum\limits_{i=1}^m [y^{(i)}log(\hat{y}^{(i)}) + (1-y^{(i)})log(1-\hat{y}^{(i)})])$
To train the parameters, we use gradient descent with the update rule:  
$w := w- \alpha \frac{d}{dw} J(w,b)$  
$b := b - \alpha \frac{d}{db} J(w,b)$  
where $\alpha$ is the learning rate which controls how big a step we take on each iteration on gradient descent.  
$\frac{d}{dw} J(w,b)$ and $\frac{d}{db} J(w,b)$ are the derivative of $J(w,b)$ which respect to $w$ and $b$ respectively.
  


