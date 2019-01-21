#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

v1 = tf.Variable(tf.constant([[3.0, 4.0],[5.0,6.0]]) )
v2 = tf.Variable(tf.constant([[7.0, 8.0],[9.0,10.0]]) )
global_step = tf.Variable(tf.constant([1]), trainable = False )

c1 = tf.global_variables()
c2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
c3 = tf.trainable_variables()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c1))
    #c2和c1输出结果一样
    print(sess.run(c2))
    print(sess.run(c3))

