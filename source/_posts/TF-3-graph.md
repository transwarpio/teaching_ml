---
title: TF-3--graph
date: 2016-12-01 19:20:49
tags:
---

[Previous Chapter: Tensorflow Basics](../TF-2-basics)
[Next Chapter: Summary and Tensorboard](../TF-4-summary)

## Graph
A **Graph** in Tensorflow represents complicated computation dataflow consisting of **Tensors**.<br>
A **Tensor** is a basic data structure in **Tensorflow**. There are several features of a **Tensor**.
- Represents one of outputs of an **Operation**;
- As a symbolic handle to one of the outputs of an **Operation**, **Tensor** provides a mean of computing the outputs in **Tensorflow** session instead of hold the real value;
- A **Tensor** could also be fed as an input to another **Operation**, that enables Tensorflow to build a multi-step, complicated computation which is called a **Graph**;
- After the **Graph** has been launched to a **Session**, the value of the **Tensor** can be computed by passing it to `Session.run()`;

### Exercise: Build a Softmax Regression in Tensorflow

#### Logistic Regression

**Logistic Regression** applies a sigmoid function on linear combination to break the constant gradient. As ranging between 0 and 1, sigmoid function is widely used in **Neural Network** for neural activation.

A sigmoid function is defined as {% raw %} $\normalsize \sigma(z) = {1 \over 1 + e^{-z}}$, where $\normalsize z = x^T * w + b$. {% endraw %}

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/2000px-Logistic-curve.svg.png" width="300" align="center"/>

#### Softmax Regression

**Logistic Regression** could properly deal with 2-class classification problem. While in machine-learned neural networks, **Softmax Regression** is more common used because of the capability of multiple-class classfiction. Generally, **Softmax Regression** is a special case of **Logistic Regression** and is designed for filling the vacancy on its disadvantages.

A Softmax function is defined as: {% raw %} $\sigma(z)_j = \Large {{e^{z_j} \over \Sigma^k_{k=1} e^{z_k}}}$ {% endraw %}

The largest $\sigma(z)_j$ is then chosen as the predicted class.

#### Relationship between Logistic Regression and Softmax Regression

Let's do some simple mathmatics.

When k = 2,
{% raw %}
$
\begin{align*} 
\sigma(z)
&= \normalsize{{1 \over e^{z_1} + e^{z_2}} \begin{bmatrix} e^{z_1} \\ e^{z_2} \end{bmatrix}} \\
&= \large\begin{bmatrix} {1 \over 1 + e^{(z_2 - z_1)}} \\ {1 \over 1 + e^{(z_1 - z_2)}}\end{bmatrix} \\
&= \large\begin{bmatrix} {1 \over 1 + e^{(z_2 - z_1)}} \\ \normalsize 1 - {1 \over 1 + e^{(z_2 - z_1)}}\end{bmatrix}
\end{align*}
$
{% endraw %}
<br>

Assume $Z = z_1 - z_2$, one of the $\sigma(z_1) = \large{1 \over 1 + e^{-Z}}$ while the other one $\sigma(z_1) = 1 - \large{1 \over 1 + e^{-Z}}$, which proves the function is consitent with *Logistic Regression*.

##### Now try to build a Softmax Regression in Tensorflow yourself. See [*Linear Regression sample*](../TF-Learn.ipynb#samplecode) for reference.

#### Necessary Headers


```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
```

#### MNIST data


```python
## MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/root/tensorflow/MNIST_data", one_hot=True)
```

#### Training Parameters


```python
## parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
```

#### Inputs


```python
## inputs
x = tf.placeholder(tf.float32, [None, 784]) # MNIST data image are of shape 28*28
y = tf.placeholder(tf.float32, [None, 10]) # MNIST data has 10 classes
```

#### Variables


```python
## variables

# initialize random uniform distributed weights with size of [784, 10] ranging from -1 to 1
W = tf.Variables(tf.random_uniform([784,10], -1.0, 1.0)) ###### write your code here ######

# initialize bias with size of [10] to zero
b = tf.Variables(tf.zeros([10])) ###### write your code here ######
```

#### Graph


```python
## graph

# comb = W * x + b (using a similar tensorflow function)
comb = ... ###### write your code here ######

# predicted value
pred = tf.nn.softmax(comb)

# entr equals to **negative** `tf.reduce_sum()` of y * log(pred), with reduction_indices = 1
entr = ... ###### write your code here ######

# cross entropy cost
cost = tf.reduce_mean(entr)

# optimizer
opti = tf.train.GradientDescentOptimizer(learning_rate)

# training_steps use optimizer to minimize the cost
training_steps = ... ###### write your code here ######

# initialization
init = tf.initialize_all_variables()
```

#### Run a Session


```python
## training
with tf.Session() as sess:
    sess.run(init)

    # training epochs
    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # split the data into different batches and run
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # run training steps and cost both in session, which should be fed `x = batch_xs` and `y = batch_ys`
            _, cur_cost = ... ###### write your code here ######

            avg_cost += cur_cost / total_batch

        # show the average cost
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```

[Previous Chapter: Tensorflow Basics](../TF-2-basics)
[Next Chapter: Summary and Tensorboard](../TF-4-summary)

