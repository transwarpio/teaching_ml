---
title: TF-4-summary
date: 2016-12-01 19:37:00
tags:
---

[Previous Chapter: Graph](3-graph.ipynb)
<br>
[Next Chapter: Neural Network](5-ann.ipynb)

## Summary

To visualize the result of a graph, *Tensorflow* introduces *summary*, which could be collected and viewed in *Tensorboard*.

<img src="https://www.tensorflow.org/versions/r0.11/images/graph_vis_animation.gif" width="800" align="left"/>


#### Import the previous softmax model


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

## MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/root/tensorflow/MNIST_data", one_hot=True)

## parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 5
```

#### Graph


```python
## inputs
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
# reshaped input image
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

## variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## graph
pred = tf.nn.softmax(tf.matmul(x,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## accuracy
tp = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(tp, tf.float32))
```

#### Summary Operations
- tf.scalar_summary()
- tf.histogram_summary()
- tf.image_summary()
- tf.audio_summary()

### Exercise: Life Cycle of Summay
- Create the TensorFlow graph that you'd like to collect summary data from, and decide which nodes you would like to annotate with summary operations;


```python
## summaries

# image summary
tf.image_summary('input', x_reshaped, 10)

# histogram summary
# create two histogram summaries here, summarizing `W` and `b`
tf.histogram_summary('weight', W)
tf.histogram_summary('bias', b)
###### write your code here ######
###### write your code here ######

# scalar summary
# create two scalar summaries here, summarizing `cost` and `accuracy`
tf.scalar_summary('cost', cost)
tf.scalar_summary('accuarcy', accuracy)
###### write your code here ######
###### write your code here ######
```

- Operations in TensorFlow don't do anything until you run them, neither do summaries. So use `tf.merge_summary` or `tf.merge_all_summaries` to combine them into a single op that generates all the summary data;


```python
# use `tf.merge_all_summaries()` to register the summaries
merged = tf.merge_all_summaries() ###### write your code here ######
```

- Create a `tf.train.SummaryWriter`;


```python
# summary writer
writer = tf.train.SummaryWriter('/root/tensorflow/summaries/softmax', graph = tf.get_default_graph())
```

- Run the merged summary op in your Session, then pass the summary protobuf to your `tf.train.SummaryWriter`;


```python
## training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # training epochs
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cur_cost, summary = sess.run([training_steps, cost, merged], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += cur_cost / total_batch

            # write your summaries
            writer.add_summary(summary, epoch * total_batch + i)

        # show the average cost and add summary
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    print("Accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
```

- Close your writer;


```python
writer.close()
```

#### Visualization
Now head to http://172.16.3.227:6006 !

#### Why my graph looks so messy?
Try to use `tf.name_scope()` wrapping your graph and re-run the **Life Cycle**.


```python
## reset the graph
tf.reset_default_graph()

## inputs
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

## weights
with tf.name_scope('weights'):
    W = tf.Variable(tf.zeros([784, 10]))

## biases
with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([10]))

## softmax
with tf.name_scope('softmax'):
    pred = tf.nn.softmax(tf.matmul(x,W) + b)
    ###### write your code here ######

## graph
with tf.name_scope('cost'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
    ###### write your code here ######

# specify optimizer
with tf.name_scope('train'):
    training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    ###### write your code here ######

## accuracy
with tf.name_scope('Accuracy'):

    tp = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(tp, tf.float32))

    ###### write your code here ######
    ###### write your code here ######
```

[Previous Chapter: Graph](3-graph.ipynb)
<br>
[Next Chapter: Neural Network](5-ann.ipynb)

