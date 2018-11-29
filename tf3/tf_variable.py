#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

v1 = tf.Variable(initial_value = [[1.0,2.0], [3.0, 4.0]], dtype=tf.float32)
v2 = tf.Variable(initial_value = tf.random_normal([2,2], stddev = 1, seed = 1), dtype=tf.float32, name = 'v2')
v21 = tf.Variable(initial_value = tf.random_normal([2,3], stddev = 1, seed = 1), dtype=tf.float32, name = 'v2')
weights = tf.Variable(initial_value = tf.random_normal([2,3], stddev = 2, seed = 1))
bias = tf.Variable(initial_value = tf.constant(0.1, shape = [1,]) )

weights1 = tf.Variable(initial_value = weights.initialized_value(), name = 'w1');

with tf.variable_scope("get_variable", reuse=tf.AUTO_REUSE):
    w1 = tf.get_variable(initializer = tf.random_normal([1,2], stddev = 1, seed = 1),  name = 'v3')
    w2 = tf.get_variable(initializer = tf.zeros(1,2),  name = 'v3')
v4 = tf.Variable([2,3], dtype=tf.float32)

original = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
z_like = tf.zeros_like(original)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1.name, '->', sess.run(v1))
    print(v2.name,'->', sess.run(v2))
    print(v21.name,'->', sess.run(v21))
    print('================')
    print(w1.name, '->', sess.run(w1))
    print(w2.name, '->', sess.run(w2))
    print('================')
    print(weights.name, '->', sess.run(weights))
    print(bias.name, '->', sess.run(bias))
    print(v4.name,'->', sess.run(v4))
    print(weights1.name,'->', sess.run(weights1))
    print(z_like.name,'->', sess.run(z_like))
