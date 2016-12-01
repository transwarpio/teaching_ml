---
title: TF-9-distributed
date: 2016-12-01 19:48:16
tags:
---

[Previous Chapter: Recurrent Neural Network](8-rnn.ipynb)

## Distributed Mode

Here is a sample of running a *Neural Network* on distributed clustering. Because of the busted Tensorflow distributed management, it could not run now here. Hope it could still help you though.

```python
"""
Here is a mnist training example using a single-layer 
neural network and a softmax classifier.
To run this file,
1. please check '172.16.3.227:~/tensorflow/scripts'
   and execute 'exec_mnist_distributed.sh'.
2. execute 'python mnist_distributed.py \
            --job_name=worker \
            --task_index=${0|1|2}' on each worker
   and 'python mnist_distributed.py \
        --job_name=ps \
        --task_index=0' on parameter server

Check (https://www.tensorflow.org/versions/r0.10/how_tos/style_guide.html) for tensorflow styling guide.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input or default parameters
flags = tf.app.flags

flags.DEFINE_string(
    "data_dir",
    "/root/tensorflow/MNIST_data",
    "Directory for storing mnist data")
flags.DEFINE_string(
    "log_dir",
    "/root/tensorflow/logs/mnist_log",
    "Directory for storing log")
flags.DEFINE_boolean(
    "download_only",
    False,
    "Only perform downloading of data")
flags.DEFINE_string(
    "job_name",
    None,
    "job name: worker or ps")
flags.DEFINE_integer(
    "task_index",
    None,
    "Worker task index, should be >= 0.")
flags.DEFINE_integer(
    "hidden_units",
    100,
    "Number of units in the hidden layer of the NN")
flags.DEFINE_integer(
    "training_steps",
    20000,
    "Number of (global) training steps to perform")
flags.DEFINE_integer(
    "batch_size",
    100,
    "Training batch size to be fetched each time")
flags.DEFINE_float(
    "learning_rate",
    0.01,
    "Learning rate in machine learning")
flags.DEFINE_string(
    "ps_hosts",
    "172.16.3.230:2222",
    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string(
    "worker_hosts",
    "172.16.3.227:2222,172.16.3.228:2222,172.16.3.229:2222",
    "Comma-separated list of hostname:port pairs")

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28


def main(_):

  # validate and print necessary arguments
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")
  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  # parse the ps(es) and worker(s)
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  num_workers = len(worker_spec)

  # construct the cluster and server
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
  server = tf.train.Server(
      cluster,
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # replica the devices
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # weight(s) and bias(es) of the hidden layer
      hid_w = tf.Variable(tf.truncated_normal(
        [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
        stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # weight(s) and bias(es) of the softmax layer
      sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
        stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      # inputs for future calculation 
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      # hidden layer computation logic
      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      # softmax layer computation logic
      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      # global step
      global_step = tf.Variable(0, name="global_step", trainable=False)

      # train step
      train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
          cross_entropy, global_step=global_step)

      # accuary calculation
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      # initialization
      init_op = tf.initialize_all_variables()
      summary_op = tf.merge_all_summaries()
      # TODO: (xiao) restore problem(temporary annotated)
      # saver = tf.train.Saver(tf.all_variables(), sharded=True)

    # create a training supervisor
    sv = tf.train.Supervisor(
        is_chief=(FLAGS.task_index == 0),
        logdir=FLAGS.log_dir,
        init_op=init_op,
        summary_op=summary_op,
	# TODO: (xiao) checkpoint cannot be restored here
	# 1. figure out how to use sharded saver
	# 2. use hdfs instead when 0.11 released
        saver=None,
        global_step=global_step,
        save_model_secs=600)

    # mnist data(exit the system is download_only is set True)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with sv.managed_session(server.target) as sess:

      time_begin = time.time()
      local_step = 0
      step = 0

      while not sv.should_stop():
        # Training feed
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        # perform training
        _, step = sess.run([train_step, global_step], feed_dict=train_feed)
        local_step += 1

        now = time.time()
        print("%f: Worker %d: training step %d done (global step: %d)"
            % (now, FLAGS.task_index, local_step, step))

        if step >= FLAGS.training_steps:
          break

      time_end = time.time()
      training_time = time_end - time_begin
      print("Training elapsed time: %f s" % training_time)

      # Validation feed
      val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
      val_xent = sess.run(cross_entropy, feed_dict=val_feed)
      print("After %d training step(s), validation cross entropy = %g"
          % (FLAGS.training_steps, val_xent))

      # let's calculate the accuracy
      pred_feed = {x: mnist.test.images, y_: mnist.test.labels}
      print("Accuracy is: %f" % sess.run(accuracy, feed_dict=pred_feed))

    # sv.stop()


if __name__ == "__main__":
  tf.app.run()

```

[Previous Chapter: Recurrent Neural Network](8-rnn.ipynb)

