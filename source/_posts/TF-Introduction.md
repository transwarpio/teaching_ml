---
title: TF-Introduction
date: 2016-12-01 19:13:38
tags:
---

## TF Playground

This is a tutorial for better understanding the concept and syntax of **Tensorflow**. In this tutorial, we assume you already have the basic knowledge of **Python** and **Machine Learning**. You can fill the blank lines, run the script and see what would happen according to the code. This tutorial will begin with some style advice and some basic concepts, following by some exercises of different **Machine Learnig** models.

###### Please feel free to add some chapters if you think it might be good for our group.

### Agenda
* Tensorflow Basics
    * [Style Guide](../TF-1-style)
    * [Tensorflow Basics](../TF-2-basics)
    * [Graph](../TF-3-graph)
    * [Summary and Tensorboard](../TF-4-summary)
* Deep Learning in TF
    * [Neural Network](../TF-5-ann)
    * [Autoencoder](../TF-6-autoencoder)
    * [Convolutional Neural Network](../TF-7-cnn)
    * [Recurrent Neural Network](../TF-8-rnn)
* Clustering
    * [Distributed Mode](../TF-9-distributed)

### Sample Code

Here is a sample **Tensorflow** code, play around and grab as much information as you can here. We'll explain it in detail in later chapters. When you think the code is ready, toggle the **"run cell"** button in the menu bar and see what will happen.


```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

## input

# randomly generate 100 instances with float type
x_data = np.random.rand(100).astype(np.float32)
# create a linear model: y = x * 0.1 + 0.3
y_data = x_data * 0.1 + 0.3


## build the graph(we already know w = 0.3 and b = 0.1 though, tensorflow will figure it out by its own)

# initialize uniform distributed weight vector
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# initialize bias to 0
b = tf.Variable(tf.zeros([1]))
# define the graph
y = W * x_data + b

# define the loss function
loss = tf.reduce_mean(tf.square(y - y_data))
# using gradient descent minimize the loss 
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


## initialization and launch the graph

# define the initialization
init = tf.initialize_all_variables()
# get a session and initialize the variables
sess = tf.Session()
sess.run(init)


## training

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print("steps: %i, weight: %f, bias: %f" % (step, sess.run(W), sess.run(b)))
```

    steps: 0, weight: 0.098388, bias: 0.440273
    steps: 20, weight: 0.081874, bias: 0.310411
    steps: 40, weight: 0.094683, bias: 0.303054
    steps: 60, weight: 0.098440, bias: 0.300896
    steps: 80, weight: 0.099542, bias: 0.300263
    steps: 100, weight: 0.099866, bias: 0.300077
    steps: 120, weight: 0.099961, bias: 0.300023
    steps: 140, weight: 0.099988, bias: 0.300007
    steps: 160, weight: 0.099997, bias: 0.300002
    steps: 180, weight: 0.099999, bias: 0.300001
    steps: 200, weight: 0.100000, bias: 0.300000


In the above sample, we defined a set of random inputs with `y = 0.1 * x + 0.3` and let our linear model `y = W * x + b` to learn the inputs in 200 training epochs. Results are printed per 20 time steps. `W` and `b` should be quite close to 0.1 and 0.3 respectively.

[Next Chapter: Style Guide](../TF-1-style)

