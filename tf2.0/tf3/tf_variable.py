#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

v1 = tf.Variable(initial_value = [[1.0,2.0], [3.0, 4.0]], dtype=tf.float32, name = 'v1')
v2 = tf.Variable(initial_value = tf.random.normal([2,2], stddev = 1, seed = 1), dtype=tf.float32, name = 'v2')
v21 = tf.Variable(initial_value = tf.random.normal([2,3], stddev = 1, seed = 1), dtype=tf.float32, name = 'v2')
weights = tf.Variable(initial_value = tf.random.normal([2,3], stddev = 2, seed = 1))
bias = tf.Variable(initial_value = tf.constant(0.1, shape = [1,]) )

weights1 = tf.Variable(initial_value = weights.initialized_value(), name = 'w1');

w1 = tf.Variable(initial_value = tf.random.normal([1,2], stddev = 1, seed = 1),  name = 'v3')
w2 = tf.Variable(initial_value = tf.zeros(1,2),  name = 'v3')
v4 = tf.Variable([2,3], dtype=tf.float32)

original = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
z_like = tf.zeros_like(original)

print(v1.name, '->', v1.read_value())
print(v1.name, '->', v1)
print(v2.name,'->', v2)
print(v21.name,'->', v21)
print('================')
print(w1.name, '->', w1)
print(w2.name, '->', w2)
print('================')
print(weights.name, '->',weights)
print(bias.name, '->', bias)
print(v4.name,'->', v4)
print(weights1.name,'->', weights1)
#print(z_like.name,'->', z_like)
print('->', z_like)
