这个文档将作为我们已经做过分享，或者将要进行分享的主题的一个索引。主要分为三个大块：常见机器学习算法，深度学习以及高级机器学习算法。常见机器学习算法主要是介绍一些spark mllib中已经实现的算法，通常要求对数学原理、mllib中的代码实现，以及如何应用于实际问题的解决等方面都要比较熟练的掌握。深度学习主要是针对一些常见的概念、优化的trick等的介绍，以及在流行深度学习框架上解决实际问题。而高级机器学习算法，通常对数学原理以及如何使用等做介绍。

以下将按照这个分类列出相应的topic。

# 常见机器学习算法
## 已分享
1. [SVM](2016/08/30/svm)
2. [关联规则](2016/07/04/关联规则挖掘基础篇)
3. [ALS](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E6%8E%A8%E8%8D%90/ALS.md)
4. [LDA](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/LDA/lda.md)
5. [Gaussian Mixture](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/gaussian-mixture/gaussian-mixture.md)
6. [Bistecting KMeans](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/bis-k-means/bisecting-k-means.md)
7. [KMeans](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/k-means/k-means.md)
8. [PIC](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E8%81%9A%E7%B1%BB/PIC/pic.md)
9. [Factor Analysis](2016/12/01/FactorAnalysis)
10. [Logistic Regression](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/logic-regression.md)
11. [Decision Tree](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E5%86%B3%E7%AD%96%E6%A0%91/decision-tree.md)
12. [Random Forest](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E7%BB%84%E5%90%88%E6%A0%91/%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97/random-forests.md)
13. [Gradient-boosted tree](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E7%BB%84%E5%90%88%E6%A0%91/%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A0%91/gbts.md)
14. [Survival Regression](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E7%94%9F%E5%AD%98%E5%9B%9E%E5%BD%92/survival-regression.md)
15. [Isotonic Regression](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92/%E4%BF%9D%E5%BA%8F%E5%9B%9E%E5%BD%92/isotonic-regression.md)
16. [BFGS and L-BFGS](https://github.com/endymecy/spark-ml-source-analysis/blob/master/%E6%9C%80%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/L-BFGS/lbfgs.md)
## 将分享

9. One-vs-Rest classifier
10. boosting and bagging
13. SGD
15. 其它最优化算法
16. 集成学习相关算法介绍

# 自然语言处理

1. [NLP介绍](2017/08/14/NLP)
2. [最大熵模型](2017/08/15/最大熵模型)
3. [Word2Vec](2017/08/11/Word2vec)
4. [分词和HMM](2017/08/16/分词和HMM)
5. [时间序列分析](2017/08/17/时间序列分析)


# 深度学习
## コースの周先生Tensorflow
#### [Introduction](2016/12/01/TF-Introduction)
1. [Style](2016/12/01/TF-1-style)
2. [Basics](2016/12/01/TF-2-basics)
3. [Graph](2016/12/01/TF-3-graph)
4. [Summary](2016/12/01/TF-4-summary)
5. [Artificial neural network](2016/12/01/TF-5-ann)
6. [Autoencoder](2016/12/01/TF-6-autoencoder)
7. [Convolution neural network](2016/12/01/TF-7-cnn)
8. [Recursive neural network](2016/12/01/TF-8-rnn)
9. [Distributed](2016/12/01/TF-9-distributed)

## 已分享
1. [MXNet框架从原理到代码](2016/07/05/mxnet)
2. [深度信念网络在蛋白质突变检测中的应用](https://github.com/xzry6/notes/blob/master/transwarp/dbn.md)
3. [RBM](https://deeplearning4j.org/cn/restrictedboltzmannmachine)
4. [DBN](https://deeplearning4j.org/cn/deepbeliefnetwork)
5. [CNN](https://deeplearning4j.org/cn/convolutionalnets)
6. [Deep auto encoder](https://deeplearning4j.org/cn/deepautoencoder)
7. [LSTM](https://deeplearning4j.org/cn/lstm)
8. [RNN](https://deeplearning4j.org/cn/usingrnns)
## 将分享
1. tensor,conv,pooling
3. BP


# 高级机器学习算法
## 已分享
1. TODO

## 将分享
6. 强化学习相关算法介绍
7. 迁移学习相关算法介绍

# 源码解析

## 将分享
1.deeplearning4j中深度学习和NLP源码分享
2.tensorflow源码分享
