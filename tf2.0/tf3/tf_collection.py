#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

v1 = tf.Variable(tf.constant([[3.0, 4.0],[5.0,6.0]]) )
v2 = tf.Variable(tf.constant([[7.0, 8.0],[9.0,10.0]]) )
global_step = tf.Variable(tf.constant([1]), trainable = False )

c3 = tf.trainable_variables()

print(v1)
#c2和c1输出结果一样
print(v2)
print(c3)

