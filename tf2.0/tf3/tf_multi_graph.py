#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

a = tf.constant([1.0, 2.0], dtype=tf.float32, name = 'a')
b = tf.constant([3.0, 4.0], dtype=tf.float32, name = 'b')
c = a + b

#定义一个计算图
g1 = tf.Graph()
#设置图为默认计算图
with g1.as_default():
    a = tf.constant([5.0, 6.0], dtype=tf.float32, name = 'a')
    b = tf.constant([7.0, 8.0], dtype=tf.float32, name = 'b')
    c1 = a + b

#设置g1运行在cpu:0上
with g1.device('/cpu:0'):
    with tf.Session(graph = g1) as sess:
        print('g1 graph:')
        print(sess.run(c1))
        print(tf.get_default_graph())

#设置默认计算图运行在gpu:0上
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True,allow_soft_placement=True)
with tf.get_default_graph().device('/gpu:0'):
    with tf.Session(config = config) as sess:
        print('default graph:')
        print(sess.run(c))
        print(tf.get_default_graph())
