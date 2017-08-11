---
title: '[Word2vec]'
date: 2017-08-11 15:38:07
tags:
---

# Word2Vec

## Outline
+ 统计语言模型
+ 神经概率语言模型
+ 基于Hierarchical Softmax的CBOW模型
+ 基于Negative Sampling的Skip-gram模型

### 统计语言模型
+ __定义__:给定一个特定顺序的词串，统计语言模型计算该词串是一个有意义的句子的概率

    p(w 1 , w 2 ,..., w t )=p(w 1 )·p(w 2 |w 1 )· ... ·p(w t |w 1 , w 2 ,..., w t-1 )
+ __例子__:

    p("Today is Friday")≈0.001 > p("Today Friday is")≈0.00000001
+ __复杂度估计__:
  - 假设词典大小为N,句子的长度为t,则共有N t 种组合。每一种组合包含t个参
数,总共需要计算和存储t·N t 个参数。

#### N-Gram语言模型
+ __基本思想__:一个词出现的概率只与其前面n-1个词相关

  p(w k |w 1 ,w 2 ...w k-1 )≈p(w k |w k-n+1 ,w k-n+2 ,...,w k-1 )

  =p(w k-n+1 ,w k-n+2 ,...,w k )/p(w k-n+1 ,w k-n+2 ,...,w k-1 )

  ≈count(w k-n+1 ,w k-n+2 ,...,w k )/count(w k-n+1 ,w k-n+2 ,...,w k-1 )
+ __平滑化问题__:
  - 若count(w k-n+1 ,w k-n+2 ,...,w k )=0,能否认为p(w k |w 1 ,w 2 ...w k-1 )=0?
  - 若count(w k-n+1 ,w k-n+2 ,...,w k-1 )=count(w k-n+1 ,w k-n+2 ,...,w k ),能否认为
p(w k |w 1 ,w 2 ...w k-1 )=1?
### 神经概率语言模型
![alt text](Word2vec/1.png)



