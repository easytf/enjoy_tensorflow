#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

v1 = tf.Variable(initial_value = [[1.0,2.0], [3.0, 4.0]], dtype=tf.float32)
v2 = tf.Variable(initial_value = tf.random_normal([2,3], stddev = 1, seed = 1), dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    print(sess.run(v2))
