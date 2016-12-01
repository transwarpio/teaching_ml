---
title: TF-7-cnn
date: 2016-12-01 19:46:57
tags:
---

[Previous Chapter: AutoEncoder](../TF-6-autoencoder)
[Next Chapter: Recurrent Neural Network](../TF-8-rnn)

## Convolutional Neural Network

In machine learning, a *Convolutional Neural Network* (CNN, or ConvNet) is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex. Individual neurons of the animal cortex are arranged in such a way that they respond to overlapping regions tiling the visual field, which can mathematically be described by a convolution operation. Convolutional networks were inspired by biological processes and are variations of multilayer perceptrons designed to use minimal amounts of preprocessing. They have wide applications in **image and video recognition**, **recommender systems** and **natural language processing**.

Following note is partially quoted from [Standford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/).

### Note: Convolutional Neural Network(CNN)

#### Structure
![cnnStructure](http://parse.ele.tue.nl/cluster/2/CNNArchitecture.jpg)

#### Convolution
Since images have the 'stationary' property, which implies that features that are useful in one region are also likely to be useful for other regions. Thus, we use a rather small patch than the image size to convolve this image. For example, when having an image with m * m * r, we could use a n * n * q patch(where n < m && q <= r). The output will be produced in size m - n + 1.

<img src="http://ufldl.stanford.edu/tutorial/images/Convolution_schematic.gif" width="440"/>)

#### Pooling
To further reduce computation, pooling is introduced in CNN. As mentioned previously, a mean pooling is applied into each region of the image because of the 'stationary' property. Pooling usually ranges from 2 to 5.

<img src="http://ufldl.stanford.edu/tutorial/images/Pooling_schematic.gif" width="500"/>)

#### Others
* Densely Connected Layers: after several convolutional layers, a few densely conncetedly layers are usually constructed before the output layer;
* Top Layer Classifier: a top classifier is used to do supervised learning on CNN;
* Back Propogation: unsample on pooling layer and use the flipped filter on convolution layer;
Here is an exercise of building a *Convolutional Neural Network* using *Tensorflow*. 

### Exercise: Convolutional Neural Network

#### Necessary Header


```python
## header
###### write your code here ######
```

#### MNIST data


```python
## MNIST data
mnist = ... ###### write your code here ######
```

#### Parameters


```python
## parameters
learning_rate = ... ###### write your code here ######
training_iters = ... ###### write your code here ######
batch_size = ... ###### write your code here ######
display_step = ... ###### write your code here ######
```

#### Inputs


```python
## placeholder inputs
x = ... ###### write your code here ######
y = ... ###### write your code here ######
# dropout is a probability for randomly dropping a unit away, it should be a float 32 value
dropout = ... ###### write your code here ######
```

#### Weights and Biases


```python
## weights and biases
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'output': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd': tf.Variable(tf.random_normal([1024])),
    'output': tf.Variable(tf.random_normal([n_classes]))
}
```

#### Graph

In a traditional *Convolutional Neural Network*, we have several *convolution layers* and *pool layers* following by a *fully-connected layer*.

###### Covolutional Layer


```python
# convolutional layer
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
```

###### Pool Layer


```python
# max pool layer
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='SAME')
```

###### Model


```python
# graph
def conv_net(x, weights, biases, dropout):
    # reshape
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # convolution layer 1
    conv1 = ... ###### write your code here ######
    # max pooling layer 1
    conv1 = ... ###### write your code here ######

    # convolution layer 2
    conv2 = ... ###### write your code here ######
    # max pooling layer 2
    conv2 = ... ###### write your code here ######

    # fully connected layer
    # reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(conv2, [-1, weights['wd'].get_shape().as_list()[0]])
    # apply `tf.nn.relu()` on linear combination of `fc * w[wd] + b[wd]`
    fc = ... ###### write your code here ######
    
    # apply dropout on fully connected layer
    fc = tf.nn.dropout(fc1, dropout)

    # output is also a linear combination
    output = ... ###### write your code here ######

    return output
```

#### Training Steps


```python
# predicted value
pred = conv_net(x, weights, biases, dropout)

# cost and optimizer
cost = ... ###### write your code here ######
optimizer = ... ###### write your code here ######
training_steps = ... ###### write your code here ######

# accuracy
correct_prediction = ... ###### write your code here ######
accuracy = ... ###### write your code here ######

# initialization
init = tf.initialize_all_variables()
```

#### Run a Session


```python
## training
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # run training steps, use `x = batch_x`, `y = batch_y` and `dropout = 0.5` here
        sess.run(... ###### write your code here ######)

        if step % display_step == 0:
            # run batch loss and accuracy, use `x = batch_x`, `y = batch_y` and `dropout = 1` here
            loss, acc = ... ###### write your code here ######

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      dropout: 1.}))
```

[Previous Chapter: AutoEncoder](../TF-6-autoencoder)
[Next Chapter: Recurrent Neural Network](../TF-8-rnn)

