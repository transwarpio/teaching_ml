---
title: TF-8-rnn
date: 2016-12-01 19:47:44
tags:
---

[Previous Chapter: Convolutional Neural Network](../TF-7-cnn)
[Next Chapter: Distributed Mode](../TF-9-distributed)

## Recurrent Neural Network and (LSTM)

Following part is an introduction to recurrent neural networks and LSTMs quoted from [this great article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

### Introduction to Recurrent Neural Network

#### Recurrent Neural Network

Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don’t throw everything away and start thinking from scratch again. *Recurrent neural networks* are deep learning networks addressing this issue. <span style="color: #F08080">They have loops in them, allowing information to persist.</span>

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" width="500"/>

This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They’re the natural architecture of neural network to use for such data. RNNs have been successfly applied to a variety of problems: <span style="color: #F08080">*speech recognition*</span>, <span style="color: #F08080">*language modeling*</span>, <span style="color: #F08080">*translation*</span>, <span style="color: #F08080">*image captioning*</span>.

#### A Major Problem in RNN

Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png" width="500" />

<span style="color: #F08080">Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.</span>

#### LSTM Network

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, <span style="color: #F08080">capable of learning long-term dependencies</span>. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to <span style="color: #F08080">avoid the long-term dependency problem</span>. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png" width="500" />

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are <span style="color: #F08080">four, interacting in a very special way</span>.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width="500" />

<span style="color: #F08080">In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others.</span> The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png" width="500" />


The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. <span style="color: #F08080">It’s very easy for information to just flow along it unchanged.</span>

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png" width="500" />

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" width="100" />

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

<span style="color: #F08080">An LSTM has three of these gates, to protect and control the cell state.</span>

#### Attention System

LSTMs were a big step in what we can accomplish with RNNs. It’s natural to wonder: is there another big step? A common opinion among researchers is: “Yes! There is a next step and it’s attention!” The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs. In fact, Xu, et al. (2015) do exactly this – it might be a fun starting point if you want to explore attention! There’s been a number of really exciting results using attention, and it seems like a lot more are around the corner…

Attention isn’t the only exciting thread in RNN research. For example, Grid LSTMs by Kalchbrenner, et al. (2015) seem extremely promising. Work using RNNs in generative models – such as Gregor, et al. (2015), Chung, et al. (2015), or Bayer & Osendorfer (2015) – also seems very interesting. The last few years have been an exciting time for recurrent neural networks, and the coming ones promise to only be more so!

### Exercise: Recurrent Neural Network

This is an exercise of *Recurrent Neural Network*.

#### Same as previous chapter


```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

## import tensorflow rnn and rnn_cell
from tensorflow.python.ops import rnn, rnn_cell

## MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/root/tensorflow/MNIST_data", one_hot=True)

## parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
```

#### A little bit difference
To classify images using a recurrent neural network, we consider every image row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.


```python
## graph parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

## inputs
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
```

#### Training Steps


```python
## predicted Value
pred = RNN(x, weights, biases)

## cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## accuracy
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

## initialization
init = tf.initialize_all_variables()
```

#### Weights and Biases


```python
## weights and biases
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}
```

#### Graph


```python
## graph
def RNN(x, weights, biases):

    # transpose batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # linear activation
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
```

#### Summary


```python
# summary writer
writer = tf.train.SummaryWriter('/root/tensorflow/summaries/rnn', graph = tf.get_default_graph())
```

#### Run a Session


```python
# training
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # keep training until reach max iterations
    while step * batch_size < training_iters:
        # reshape data to get 28 sequence of 28 elements
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
```

[Previous Chapter: Convolutional Neural Network](../TF-7-cnn)
[Next Chapter: Distributed Mode](../TF-9-distributed)

