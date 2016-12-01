---
title: TF-6-autoencoder
date: 2016-12-01 19:46:18
tags:
---

[Previous Chapter: Neural Network](5-ann.ipynb)
<br>
[Next Chapter: Convolutional Neural Network](7-cnn.ipynb)

## AutoEncoder

An **Autoencoder** is a deep learning algorithm which has two parts, <span style="color: #F08080">*encoder*</span> and <span style="color: #F08080">*decoder*</span>. An *autoencoder* is very similar to an *artificial neural network* other than <span style="color: #F08080">using inputs as output labels to optimize the parameters</span>, which makes it quite like an unsupervised model. Well, it is **NOT￼** though.

![](https://blog.keras.io/img/ae/autoencoder_schema.jpg)

The following explanation of *Autoencoder* is quoted from [Keras](https://blog.keras.io/building-autoencoders-in-keras.html).

The **encoder** and **decoder** in Autoencoder are <span style="color: #F08080">1) data-specific</span>, <span style="color: #F08080">2) lossy</span>, and <span style="color: #F08080">3) learned automatically from examples rather than engineered by a human</span>. Additionally, in almost all contexts where the term "autoencoder" is used, the compression and decompression functions are implemented with neural networks.

1. Autoencoders are data-specific, which means that they will only be able to compress data similar to what they have been trained on. This is different from, say, the MPEG-2 Audio Layer III (MP3) compression algorithm, which only holds assumptions about "sound" in general, but not about specific types of sounds. An autoencoder trained on pictures of faces would do a rather poor job of compressing pictures of trees, because the features it would learn would be face-specific.

2. Autoencoders are lossy, which means that the decompressed outputs will be degraded compared to the original inputs (similar to MP3 or JPEG compression). This differs from lossless arithmetic compression.

3. Autoencoders are learned automatically from data examples, which is a useful property: it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input. It doesn't require any new engineering, just appropriate training data.

To build an autoencoder, you need three things: an encoding function, a decoding function, and a distance function between the amount of information loss between the compressed representation of your data and the decompressed representation (i.e. a "loss" function). The encoder and decoder will be chosen to be parametric functions (typically neural networks), and to be differentiable with respect to the distance function, so the parameters of the encoding/decoding functions can be optimize to minimize the reconstruction loss, using Stochastic Gradient Descent. It's simple! And you don't even need to understand any of these words to start using autoencoders in practice.


#### Are they good at data compression?

<span style="color: #F08080">Usually, not really.</span> In picture compression for instance, it is pretty difficult to train an autoencoder that does a better job than a basic algorithm like JPEG, and typically the only way it can be achieved is by restricting yourself to a very specific type of picture (e.g. one for which JPEG does not do a good job). The fact that autoencoders are data-specific makes them generally impractical for real-world data compression problems: you can only use them on data that is similar to what they were trained on, and making them more general thus requires lots of training data. But future advances might change this, who knows.

#### What are autoencoders good for?

They are rarely used in practical applications. In 2012 they briefly found an application in [greedy layer-wise pretraining for deep convolutional neural networks](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf), but this quickly fell out of fashion as we started realizing that better random weight initialization schemes were sufficient for training deep networks from scratch. In 2014, [batch normalization](https://arxiv.org/abs/1502.03167) started allowing for even deeper networks, and from late 2015 we could train [arbitrarily deep networks from scratch using residual learning](https://arxiv.org/abs/1512.03385).

Today two interesting practical applications of autoencoders are <span style="color: #F08080">**data denoising**</span> (which we feature later in this post), and <span style="color: #F08080">**dimensionality reduction for data visualization**</span>. With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.

For 2D visualization specifically, t-SNE (pronounced "tee-snee") is probably the best algorithm around, but it typically requires relatively low-dimensional data. So a good strategy for visualizing similarity relationships in high-dimensional data is to start by using an autoencoder to compress your data into a low-dimensional space (e.g. 32 dimensional), then use t-SNE for mapping the compressed data to a 2D plane.

#### So what's the big deal with autoencoders?

Their main claim to fame comes from being featured in many introductory machine learning classes available online. As a result, a lot of newcomers to the field absolutely love autoencoders and can't get enough of them. This is the reason why this tutorial exists!

Otherwise, one reason why they have attracted so much research and attention is because they <span style="color: #F08080">**have long been thought to be a potential avenue for solving the problem of unsupervised learning**</span>, i.e. the learning of useful representations without the need for labels. Then again, autoencoders are not a true unsupervised learning technique (which would imply a different learning process altogether), they are a <span style="color: #F08080">**self-supervised technique**</span>, a specific instance of supervised learning where the targets are generated from the input data. In order to get self-supervised models to learn interesting features, you have to come up with an interesting synthetic target and loss function, and that's where problems arise: merely learning to reconstruct your input in minute detail might not be the right choice here. At this point there is significant evidence that focusing on the reconstruction of a picture at the pixel level, for instance, is not conductive to learning interesting, abstract features of the kind that label-supervized learning induces (where targets are fairly abstract concepts "invented" by humans such as "dog", "car"...). In fact, one may argue that the best features in this regard are those that are the worst at exact input reconstruction while achieving high performance on the main task that you are interested in (classification, localization, etc).

In self-supervized learning applied to vision, a potentially fruitful alternative to autoencoder-style input reconstruction is the use of toy tasks such as jigsaw puzzle solving, or detail-context matching (being able to match high-resolution but small patches of pictures with low-resolution versions of the pictures they are extracted from). The following paper investigates jigsaw puzzle solving and makes for a very interesting read: Noroozi and Favaro (2016) Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. Such tasks are providing the model with built-in assumptions about the input data which are missing in traditional autoencoders, such as "visual macro-structure matters more than pixel-level details".

![](https://blog.keras.io/img/ae/jigsaw-puzzle.png)

### Let's begin with a common Autoencoder

#### Prerequisite
First we *import packages*, *read datasets* and *assign model parameters*.


```python
## matplotlib will be used to reconstruct the test 
%matplotlib notebook

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

## MNIST_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/root/tensorflow/MNIST_data', one_hot=True)

## parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1
testing_number = 5

## number of hidden units
h_0 = 784
h_1 = 100
h_2 = 10
```

#### Inputs and Variables
Mention that *labels* are not required here because an *autoencoder* uses inputs as output labels


```python
## inputs
x = tf.placeholder(tf.float32, [None, h_0])

## variables
w = {
    'e_1': tf.Variable(tf.random_normal([h_0, h_1])),
    'e_2': tf.Variable(tf.random_normal([h_1, h_2])),
    'd_1': tf.Variable(tf.random_normal([h_2, h_1])),
    'd_2': tf.Variable(tf.random_normal([h_1, h_0]))
}

b = {
    'e_1': tf.Variable(tf.random_normal([h_1])),
    'e_2': tf.Variable(tf.random_normal([h_2])),
    'd_1': tf.Variable(tf.random_normal([h_1])),
    'd_2': tf.Variable(tf.random_normal([h_0]))
}
```

#### Graph
Define the graph and optimizer.


```python
## graph
def feed_forward(x, layer):
    return tf.nn.sigmoid(tf.add(tf.matmul(x, w[layer]), b[layer]))

e1 = feed_forward(x, 'e_1')
e2 = feed_forward(e1, 'e_2')
d1 = feed_forward(e2, 'd_1')
d2 = feed_forward(d1, 'd_2')

mean_err = tf.reduce_mean(tf.pow(d2 - x, 2))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(mean_err)
```

#### Start a Session
Here we use *[matplotlib](http://matplotlib.org/)* to reconstruct the image.


```python
## start a session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    # train
    for epoch in range(training_epochs):
        avg_err = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, err = sess.run([optimizer, mean_err], feed_dict={x: batch_x})
            avg_err += err / total_batch

        if (epoch + 1) % display_step == 0:
            print('step: %d, average mean error is %f' % (epoch + 1, avg_err))
    
    print('optimization finished')

    # test
    encode_decode = sess.run(d2, feed_dict={x: mnist.test.images[:testing_number]})
    
    # visualize
    f, a = plt.subplots(testing_number, 2, figsize=(5, 8))
    for i in range(testing_number):
        a[i][0].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[i][1].imshow(np.reshape(encode_decode[i], (28, 28)))
            
```

[Previous Chapter: Neural Network](5-ann.ipynb)
<br>
[Next Chapter: Convolutional Neural Network](7-cnn.ipynb)

