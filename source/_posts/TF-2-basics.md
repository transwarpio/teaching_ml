---
title: TF-2-basics
date: 2016-12-01 19:19:30
tags:
---

[Previous Chapter: Style Guide](1-style.ipynb)
<br>
[Next Chapter: Graph](3-graph.ipynb)

## Basic Concpts

Following part is quoted from [Tensorflow 2015 White Paper](http://download.tensorflow.org/paper/whitepaper2015.pdf).

<span style="color: #F08080">A **TensorFlow computation** is described by a **directed graph**, which is composed of a set of nodes</span>. The *graph* represents a dataflow computation, with extensions for allowing some kinds of nodes to maintain and update persistent state and for branching and looping control structures within the graph. Clients typically construct a computational graph using one of the supported frontend languages (C++ or
Python).

<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/3.png" width="200" />
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/2.png" width="800" />

In a *TensorFlow graph*, <span style="color: #F08080">each **node** represents the instantiation of an **operation**, and has zero or more inputs and zero or more outputs</span>. <span style="color: #F08080">Values that flow along normal edges in the graph (from outputs to inputs) are **tensors**</span>, arbitrary dimensionality arrays where the underlying element type is specified or inferred at graph-construction time.

### Tensors
<span style="color: #F08080">A tensor is a typed, multidimensional array</span>. Tensorflow support a variety of tensor element types, including signed and unsigned integers ranging in size from 8 bits to 64 bits, IEEE float and double types, a complex number type, and a string type (an arbitrary byte array). Backing store of the appropriate size is managed by an allocator that is specific to the device on which the tensor resides. Tensor backing store buffers are reference counted and are deallocated when no references remain.


### Variables
In most computations a graph is executed multiple times. Most **tensors** do not survive past a single execution of the graph. However, <span style="color: #F08080">a **Variable** is a special kind of operation that returns a handle to a persistent mutable tensor that survives across executions of a graph</span>. Handles to these persistent mutable tensors can be passed to a handful of special operations, such as Assign and AssignAdd (equivalent to +=) that mutate the referenced tensor.

### Operations(Ops)
<span style="color: #F08080">An **operation** has a name and represents an abstract computation (see the following table)</span>. An operation can have attributes, and all attributes must be provided or inferred at graph-construction time in order to instantiate a node to perform the operation.
<span style="color: #F08080">A **kernel** is a particular implementation of an operation that can be run on a particular type of device (e.g., CPU or GPU)</span>.

<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs2016/imgs_tensorflow/1.png" width="800" />

### Sessions

<span style="color: #F08080">Clients programs interact with the TensorFlow system by creating a **Session**</span>.

**Session Run**, which takes a set of output names that need to be computed, as well as an optional set of tensors to be fed into the graph in place of certain outputs of nodes. Using the arguments to Run, the TensorFlow implementation can compute the transitive closure of all nodes that must be executed in order to compute the outputs that were requested, and can then arrange to execute the appropriate nodes in an order that respects their dependencies. <span style="color: #F08080">Most of our uses of TensorFlow set up a Session with a graph once, and then execute the full graph or a few distinct subgraphs thousands or millions of times via **Run** calls.</span>

### Single Device Execution
Let’s first consider the simplest execution scenario: a single worker process with a single device. The nodes of the graph are executed in an order that respects the dependencies between nodes. In particular, <span style="color: #F08080">Tensorflow keeps track of a count per node of the number of dependencies of that node that have not yet been executed. Once this count drops to zero, the node is eligible for execution and is added to a ready queue</span>. The ready queue is processed in some unspecified order, delegating execution of the kernel for a node to the device object. When a node has finished executing, the counts of all nodes that depend on the completed node are decremented.


### Exercise: Basic Concepts

Read the below **Tensorflow API** carefully and finish the exerceses.

##### tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
```
Runs operations and evaluates tensors in fetches.

The fetches argument can be one of the following types:
- An Operation;
- A Tensor;
- A SparseTensor;
- A get_tensor_handle op;
- A string which is the name of a tensor or operation in graph;

Returns
- A single value if fetches is a single graph element;
- A list of values if fetches is a list;
- A dictionary with the same keys;
```

##### tf.constant(value, dtype=None, shape=None, name='Const')
```
Creates a constant tensor.

The argument value can be
- A constant value;
- A list of values of type dtype; 

The resulting tensor is populated with values of type dtype.
```

##### tf.placeholder(dtype, shape=None, name=None)
```
Inserts a placeholder for a tensor that will be always fed.

Important: This tensor will produce an error if evaluated.
Its value must be fed using the feed_dict optional argument
to Session.run(), Tensor.eval(), or Operation.run().
```

##### tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
```
Multiplies matrix a by matrix b, producing a * b.

The inputs must be
- Two-dimensional matching matrices;
- Both matrices must be of the same type
  (float32, float64, int32, complex64)

Either matrix can be transposed on the fly by setting the
corresponding flag to True.

If one or both of the matrices contain a lot of zeros, please
set the corresponding a_is_sparse or b_is_sparse flag to True.
```


#### Prepare for Coding
At first, let's follow the [style guide](1-style.ipynb) and put the necessary headers.


```python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
```

Get a session to run the code.


```python
sess = tf.Session()
```

#### Constants


```python
## create two variables `a = 2` and `b = 3`
a = ... ###### write your code here ######
b = ... ###### write your code here ######

resultA = sess.run(a)
resultB = sess.run(b)

# see the result and its type
print("a = %i with tpye of %s" % (resultA, type(resultA)))
print("b = %i with tpye of %s" % (resultB, type(resultB)))
```

#### Arrays and Matrices


```python
## create two matrixes `c = {{1, 2}, {3, 4}}` and `d = {{1, 1}, {0, 1}}`
c = ... ###### write your code here ######
d = ... ###### write your code here ######

resultC = sess.run(c)
resultCD = sess.run([c, d])

# see the result and its type
print(resultC)
print(type(resultC))
print(resultCD)
print(type(resultCD))
```

#### String


```python
## create a string `e = 'Hello, Tensorflow'`
e = ... ###### write your code here ######
print(sess.run(e))
```

#### Matrice Multipulation


```python
## multply c and d using `tf.matmul()`
mul = ... ###### write your code here ######
print(sess.run(mul))
```

#### Placeholders


```python
## create two placeholder `f` and `g` holds `tf.int16`
f = ... ###### write your code here ######
g = ... ###### write your code here ######

## some operations
add = tf.add(f, g)
mul = tf.mul(f, g)

# get the result of add by feeding `{f: 2, g: 3}` to `sess.run()` using feed_dict
resultAdd, resultMul = ... ###### write your code here ######
print("adding result is: %i" % resultAdd)
print("multiplying result is: %i" % resultMul)
```

[Previous Chapter: Style Guide](1-style.ipynb)
<br>
[Next Chapter: Graph](3-graph.ipynb)

