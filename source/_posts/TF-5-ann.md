---
title: TF-5-ann
date: 2016-12-01 19:37:53
tags:
---

[Previous Chapter: Summary and Tensorboard](../TF-4-summary)
[Next Chapter: AutoEncoder](../TF-6-autoencoder)

## Neural Network

Here is an exercise of building a *Neural Network* using *Tensorflow*.

Following note of *Neural Network* is quoted from [here](https://github.com/xzry6/notes/blob/master/transwarp/ann.md).

### Note: 人工神经网络
**人工神经网络**是受到了生物学上动物的中枢神经系统的启发而开发出来的一种计算模型。
**人工神经网络**最早于20世纪40年代提出，但由于庞大的神经网络和当时计算能力的局限，神经网络的研究一直停滞不前。
直到20世纪80年代**分散式并行处理**的流行和由*Paul Werbos*创造的反向传播算法，神经网络渐渐又开始流行起来。并于21世纪开始同**深度学习**一起成为机器学习领域的热点。

#### 结构
结构上，人工神经网络由一个*输入可见层*，多个*隐藏层*和一个*分类输出层*组成。每一层由不同数目的*神经单元*组成, 前后两层之间的*weight*和*bias*组成了整个模型。
人工神经网络模拟了人脑神经元传递的过程，每一层的神经元都对应着通过观察而解析出的不同程度的特征。可以用一个简单的例子来理解人工神经网络的结构。
当我们观察一辆车的时候，我们首先观察到的可能是“车高”，“车宽”，“四门”，“四驱”， “疝气大灯”等一系列可见特征，对应了神经网络的可见层。之后神经网络通过可见特征对隐藏特征进行解析，于是在第一个隐藏层中我们得到了我们未观察到的信息“奥迪”，“SUV”。之后逐层解析，通过第一个隐藏层特征，我们在第二个隐藏层可能会得到“Q7”这样的特征，神经网络在不同的隐藏层会解析出不同程度的隐藏特征。当得到了这些比可见特征更加具象并且有意义的特征后，位于神经网络顶层的分类器会更加容易判断出“开车人的职业”。

<img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" width="440">

在上图中，我们可以看到输入层有三个神经元，第一和第二隐藏层分别含有四个神经元，最后输出一个神经元。同一层的神经元相互之间没有连接，代表了同一层之间的特征相互独立。相邻两层的神经元相互连接，由直线表示，被成为*weights(权重)*。每一个神经元由**所有**上一层的*神经元*和*权重*计算得出，所以只有计算出同一层所有神经元的值后，才能继续向前传递。


#### 人工神经网络的训练
人工神经网络的训练可以分为两个部分，*前向传播*和*反向传播*。*前向传播*负责逐层传递并激活神经元，并于最后分类层预测结果。*反向传播*通过计算顶层预测结果和实际结果的误差，将误差逐层传递回模型，使用梯度下降或其它方式更新模型权重。<br><br>
接下去我们会分开对*前向传播*和*反向传播*进行介绍。

##### 前向传播
对人工神经网络进行训练时，我们首先把输入放入输入可见层，*喂*进神经网络。上文已经说过，由于每一个神经元与上一层所有神经元有联系，所以人工神经网络的传递方式是**逐层**传递的。传递过程在生物意义上意味着**激活**，所以传递时用到的函数被称作激活函数。

<img src="http://ufldl.stanford.edu/tutorial/images/Network331.png" width="440">

##### 组合函数
在把参数传递进激活函数前，首先将上一层的神经元和相关权重进行组合，再加上偏置。这样的组合函数表示每一层的神经单元由上一层的单元和权重生成.

{% raw %}
传递函数: $p(x_j^n) = \sigma(\Sigma_i w_{ij}^{{n-1}n}x_i^{n-1})$
{% endraw %}

<img src="http://ufldl.stanford.edu/tutorial/images/SingleNeuron.png" width="240">

{% raw %}
上述公式中，$p(x_j^n)$表示第n层第j个神经元被激活的概率，${x_i^{n-1}}$表示第(n-1)层第i个神经元的值，$w_{ij}^{{n-1}n}$表示第(n-1)层的第i个神经元与第n层第j个神经元之间的连线权重，(n-1)层最后一个神经元+1指的是n层第j个结点的权重。
{% endraw %}

#### 激活函数
当前一层神经元和对应权重进行组合后，我们可以直接把得到的值当作当前单元的激活函数，可是由于是简单的线性函数，所以容易造成值过大和过小的两极化分布。为此，研究者们引入了一些**激活函数**来改善分布，更好地*激活*神经元。

- {% raw %}Sigmoid: $\sigma(z) = \large{1 \over 1 + e^{-z}}${% endraw %}


- {% raw %}Tanh: $\sigma(z) = \large{sinh(z) \over cosh(z)} = {{e^z - e^{-z}} \over {e^z + e^{-z}}}${% endraw %}

- ReLU: $\sigma(z) = max(0, z)$

sigmoid和tanh由于有各自的区间（sigmoid: (0, 1)，tanh: (-1, 1)），能很好的把激活值限制在这些区间内，稳定的区间同时也说明神经元之间的权重会更加稳定。但是，这两个函数在极限值造成平滑的梯度，会丢失一部分的信息。relu函数可以保留着一部分梯度，同时$max(0, z)$也会稀疏出现的negative值。

### 反向传播
反向传播是神经网络更新权重的过程，因为多层的结构，当进行迭代更新的时候，输出层产生的error会反向传遍整个网络，每一层的权重会根据误差进行更新。和一般分类器一样，神经网络顶层的误差就是分类器的误差，即预测值和实际值的误差。之后，同前向传播一样，每前一层的神经元的误差由后一层的所有神经元和误差计算得出，反向逐层传递。当误差传到底层，即所有误差都被计算出后，我们再次*前向*传播更新权重。

<img src="http://i.stack.imgur.com/H1KsG.png" width="440">

- {% raw %}输出层的error就是分类器的error: $\delta_i^n = \sigma_i^n - y_i${% endraw %}


- {% raw %}前一层的error由后一层的error产生: $\delta_i^n = \Sigma_j w_{ij}^{n+1} \delta_j^{n+1}${% endraw %}


- {% raw %}更新权重使用梯度下降: $\Delta w_{ij} = -\gamma \sigma_i^n \delta_j^{n+1}${% endraw %}

### Exercise: Neural Network
In this exercise, our neural network will have 2 hidden layer with user defined units and one linear regression output layer.

#### Same as previous chapter


```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

## MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/root/tensorflow/MNIST_data", one_hot=True)

## parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

## inputs
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
```

#### Hidden Layers


```python
## hidden layer number of features
# define any number of features to this two hidden layers
n_hidden_1 = ... ###### write your code here ######
n_hidden_2 = ... ###### write your code here ######
```

#### Weights and Biases


```python
## weights and biases
weights = {
    'w1': tf.Variable(tf.random_normal([784, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'output': tf.Variable(tf.random_normal([n_hidden_2, 10]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'output': tf.Variable(tf.random_normal([10]))
}
```

#### Define a deep graph function


```python
## graph
def multilayer_perceptron(x, weights, biases):
    # hidden layer 1
    comb_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    # layer_1 is activated using `tf.nn.relu()`
    h_1 = ... ###### write your code here ######
    
    # hidden layer 2
    comb_2 = tf.add(tf.matmul(h_1, weights['w2']), biases['b2'])
    # layer_2 is activated using `tf.nn.relu()`
    h_2 = ... ###### write your code here ######
    
    # output is just a linear combination function with `h_2 * w[output] + b[output]`
    output = ... ###### write your code here ######

    return output
```

#### Training Steps


```python
# predicted value
pred = multilayer_perceptron(x, weights, biases)

# define your cost here, use `tf.nn.softmax_cross_entropy_with_logits(pred, y)` to calculate entropy here
cost = ... ###### write your code here ######

# define your optimizer here, we use a `tf.train.AdamOptimizer()` here
optimizer = ... ###### write your code here ######

# training steps
training_steps = optimizer.minimize(cost)

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
            _, cur_cost = sess.run([training_steps, cost],
                            feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += cur_cost / total_batch

        # show the average cost
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # calculate accuracy
    # use `tf.equal()` and `tf.argmax()` to get correction prediction
    correct_prediction = ... ###### write your code here ######
    # use `tf.reduce_mean` to get accuracy here
    accuracy = ... ###### write your code here
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```

[Previous Chapter: Summary and Tensorboard](../TF-4-summary)
[Next Chapter: AutoEncoder](../TF-6-autoencoder)

