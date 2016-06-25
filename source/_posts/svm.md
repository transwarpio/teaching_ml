---
title:  suport vector machine
---

- [Introduction](#sec-1)
- [Spark](#sec-2)
  - [RDD](#sec-2-1)
  - [Cluster mode](#sec-2-2)
- [SVM](#sec-3)
  - [Introduction](#sec-3-1)
  - [SGD](#sec-3-2)
  - [Pegasos and MLlib implementation](#sec-3-3)
- [SGD in Spark](#sec-4)
  - [treeAggregate](#sec-4-1)
  - [Implementation](#sec-4-2)
- [Experiments and Performance](#sec-5)
  - [Experiments with small dataset](#sec-5-1)
  - [The convergence speed](#sec-5-2)

# Introduction<a id="orgheadline1"></a>

Spark is a fast and general engine for large-scale data processing. Spark is often compared with Hadoop. It claims that it is 100x faster than Hadoop MapReduce when computing in memory and 10x faster on disk. This speed is achieved because Spark handles computing in memory. With Hadoop MapReduce, the map reduce result is stored in disk and you have to read the result again from disk if you need it.
 But with Spark, the result is still in memory, which makes it every useful for iterative jobs that need to run map reduce many times.

And we know that iteration is very common in machine learning algorithms, such as gradient descent. So we think spark
would be a good platform for large scale machine learning and we want to test its performence.

So with the out-standing performance of Spark, in this project we'd like to analyze the performance of machine learning tasks with Spark under TeraLab which is a Big Data analysis platform built by L'Institut Mines-Télécom and Le GENES, le Groupe des Ecoles nationales d’économie et de statistique.

# Spark<a id="orgheadline4"></a>

## RDD<a id="orgheadline2"></a>

The most important data structure of Spark is **RDD(Resilient Distributed Datasets)**, a distributed memory abstraction that we
can perform in-memory operations on large clusters in a fault-tolerant manner.
Every RDD has multiple partitions which are distributed among clusters. And these partitions are the computing unit.

There are 2 main types of operations: **transformation** and **action**.
**Transformation** transforms one RDD from anthor one, such as *map* and *filter*.
**Action** get results from RDD, such as *collect* and *reduce*.
And one most important feature of RDD is the all the operations are lazy evaluated. That is to say that transformations won't be executed
only if the final action is met.

Figure [9](#orgparagraph1) shows a simple graph of operations. *A* is the initial RDD which might be read from text file. Then transforms transform *A* to *B* and *C*. Finally, an action gets a normal Scala object such as *float*, *int* or *array* from *C*.

![img](./imgs/rdd.png "RDD transformations and actions")

## Cluster mode<a id="orgheadline3"></a>

Figure [11](#orgparagraph2) shows the basic architecture of Spark running in the cluster. In the cluster mode, a Spark application consists of a single **driver** process and a set of **executor** processes scattered across nodes on the cluster. The driver is the process that is in charge of the high-level control flow of work that needs to be done. The executor processes are responsible for executing this work, in the form of tasks, as well as for storing any data that the user chooses to cache.

![img](./imgs/spark.png "Spark cluster architecture(from ![img](//spark.apache.org/docs/latest/img/cluster-overview.png))")

At the top of the execution hierarchy are **jobs**.
An action(collect, reduce&#x2026;) triggers the launch of a **job**.
Then, Spark examines the graph of RDDs on which the action depends. Spark will find the farthest back RDDs that depend on no other RDDs or already cached. After finding the dependency, Spark puts the job's transformations into **stage**. Each stage includes a colllection of **tasks** that run the same code on each subset of data.

\newpage

# SVM<a id="orgheadline8"></a>

## Introduction<a id="orgheadline5"></a>

Support Vector Machine is an effective and popular classification learning tool.

Given a training set \\( S = \{ (x_i, y_i) \}_{i=1}^{m} \\), where \\( x_i \in \mathbb{R}^n \\) and \\( y_i \in \{ +1, -1 \} \\), figure [22](#orgparagraph3) shows what a SVM looks like. We note  \\( w \cdot x - b = 0 \\) as the hyperplane where \\( w \\) is the vector to the hyperplane. What we want is to find the maximum-margin hyperplane that divides the points having \\( y_i=1 \\) from those having \\( y_i=-1 \\). That means \\( y_i(w \cdot x_i -b ) \geq 1 \\) for all \\( 1 \leq i \leq n \\).

So the optimization problem can be written as

Maximize

\\[ \frac{2}{\|w\|} \\]

Which equals to minimize

\\[ \frac{1}{2} \| w \|^2 \\]

subject to \\( y_i(w \cdot x_i - b) \geq 1 \\) for all \\( 1 \leq i \leq n \\)

![img](./imgs/svm.png "SVM")

In fact, it can be viewed as an unconstrained empirical loss minimization with a penalty term for the norm of the classifier that is being learned. We would like to find the minimization of the problem

\\[
 \min_{w} \frac{\lambda}{2}\|w\|^2 + \frac{1}{m}\sum_{(x, y) \in S} \ell(w; (x, y))
\\]

Where &lambda; is the regularization parameter, \( \ell(w, (x, y)) \) is the hinge loss:

\\[
\ell(w, (x, y)) = max\{0, 1-y \langle w, x \rangle \}
\\]

To solve this optimization problem, we can use gradient descent to achieve the minimum value.

The objective function is

\\[
f(w) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{m}\sum_{(x_i, y_i) \in S}\ell(w; (x_i, y_i))
\\]

Then, the sub-gradient for iteration *t* is
\\[
\nabla_t = \lambda w_t - \frac{1}{m}\sum_{(x_i, y_i) \in S}\mathbbm{1}[y_i \langle w, x_i \rangle < 1]y_i x_i
\\]

Now we can update \\( w \\), where \\( \eta_t \\) is the step size
\\[
w_{t+1} \leftarrow w_t - \eta_t\nabla_t
\\]

## SGD<a id="orgheadline6"></a>

From the previous section, we notice that we need to iterate all the data point when calculating gradient.
And this might be computing expensive if we have tons of data. This is the reason why Stochastic Gradient Descent(SGD) becomes so useful.
When handling large scale problems, SGD uses sub dataset at each iteration instead of the whole dataset.

So now, the object function becomes:
\\[
f(w, A_t) = \frac{\lambda}{2}\|w\|^2 + \frac{1}{k}\sum_{(x_i, y_i) \in A_t}\ell(w; (x_i, y_i))
\\]
where \( A_t \subset S \), \( |A_t| = k \). At each iteration, we takes a subset of data point.

And sub-gradient is
 \\[ \nabla_t = \lambda w_t - \frac{1}{k}\sum_{(x_i, y_i) \in A_t}\mathbbm{1}[y_i \langle w, x_i \rangle < 1]y_i x_i \\]

## Pegasos and MLlib implementation<a id="orgheadline7"></a>

Pegasos, is a kind of stochastic gradient descent algorithm. And Spark MLlib also provides an SGD implementation for us. After reading the code of MLLib, we notice that the only difference between Pegasos and MLlib is the choice of update step.

\\[
w_{t+1} \leftarrow w_t - \eta_t\nabla_t
\\]

In Pegasos, the update step is
\\[
\eta_t = \frac{\alpha}{t\lambda}
\\]

In MLlib, this is
\\[
\eta_t = \frac{\alpha}{\sqrt{t}}
\\]

where &alpha; is the step size parameter

# SGD in Spark<a id="orgheadline11"></a>

## treeAggregate<a id="orgheadline9"></a>

The main usage of Spark for SGD is to calculate the gradient which need to sum up the value of every data point. And in Spark, this is done by the RDD method **treeAggregate**. **Aggregate** is a generalized combination of **Map** and **Reduce**. The definition of **treeAggregate** is

```scala
RDD.treeAggregate(zeroValue: U)(
      seqOp: (U, T) => U,
      combOp: (U, U) => U,
      depth: Int = 2): U
```

In this method, there are three parameters in which the first two are more important for us.

-   seqOp: calculate sub gradient for every partition
-   combOp: combine the result of seqOp or upper level combOp together
-   depth: control the depth of the aggregation tree

![img](./imgs/tree.png "tree aggregate")

We can see from the figure [46](#orgparagraph4) that the first thing is to use the **seqOp** to calculate the sub gradient for every partition, then it sums them up level by level using **combOp**.

## Implementation<a id="orgheadline10"></a>

In this section, the code for the main SGD logic is shown in [2](#orgsrcblock1). It runs `numIterations` times to get the final \( w \).

First, `data.sample` takes a subset of data whose size is decided by `miniBatchFraction`. Then we use `treeAggregate` method on this sample. In `seqOp`, The `gradientSum` is updated by `axpy(y, b_x, c._1)` if \( y\langle w, x \rangle < 1 \) which means wrong classification. In `combOp`, `gradientSum` is combined together by `c1._1 += c2._1`. After we get the `gradientSum`, we calcuate `step` and `gradient`. Finally, we update the weights with `axpy(-step, gradient, weights)`

```scala
for (i <- 1 to numIterations) {
      val bcWeights = sc.broadcast(weights)

  val (gradientSum, lossSum, batchSize) = data.sample(false,
    miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](weights.size), 0.0, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val y = v.label
            val x = v.features
            val b_x = BDV(x.toArray)
            val dotProduct = bcWeights.value.dot(b_x)
            if (y * dotProduct < 1) {
              axpy(y, b_x, c._1)    // add to gradientSum
            }
            (c._1, c._2 + math.max(0, 1 - y * dotProduct), c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      val step = stepSize / (regParam * i)
      val gradient = weights * regParam - gradientSum / batchSize.toDouble
      axpy(-step, gradient, weights)    // update weights
    }
```

\newpage

# Experiments and Performance<a id="orgheadline15"></a>

## Experiments with small dataset<a id="orgheadline12"></a>

The first thing we need to do is to show that our Pegasos implementation works correctly. To achieve that, we simulate some sample 2D and 3D data with normal distribution.
The first one is a 2D linear dataset. The result is in figure [52](#orgparagraph5).

![img](./imgs/2d_linear.png "2D linear")

Then this figure [54](#orgparagraph6) shows the result of a 3D linear dataset.

![img](./imgs/3d_linear.png "3D linear")

From those experiments, we show that our Pegasos implementation works well. Then we will test it's performance with larger dataset.

## The convergence speed<a id="orgheadline13"></a>

Since the implementation of Pegasos and MLlib is slightly different. we would like to compare the convergence speed of Pegasos and MLlib. In this test, we take 5GB data with 1000 features, launch the job with 4 executors and run 100 iterations. The result is in figure [57](#orgparagraph7), where the Y axis is not aligned. In the plot, the first 30 iterations are ignored since the initial loss is too high.

<./imgs/step1.eps>

Then we align the Y axis in figure [59](#orgparagraph8). From those 2 figures, we can see that when the step size is well chosen, Pegasos and MLlib have similar performance. But Pegasos has one advantage that it is easier to find the right step parameter. In most cases, 1 is good for Pegasos.

<./imgs/step2.eps>

# Reference<a id="orgheadline18"></a>

1.  Zaharia, Matei, et al. "Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing." Proceedings of the 9th USENIX conference on Networked Systems Design and Implementation. USENIX Association, 2012
2.  Zaharia, Matei, et al. "Spark: cluster computing with working sets." Proceedings of the 2nd USENIX conference on Hot topics in cloud computing. Vol. 10. 2010
3.  Shalev-Shwartz, Shai, et al. "Pegasos: Primal estimated sub-gradient solver for svm." Mathematical programming 127.1 (2011): 3-30
