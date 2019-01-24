#!/usr/bin/python3
# -*- coding: utf-8 -*-)
import tensorflow as tf

a = tf.constant('hello tensor')
print('a:',a)
#输出a: tf.Tensor(b'hello tensor', shape=(), dtype=string)
d = tf.constant([2.0], dtype=tf.float32, name='d')
print('d:',d)
#输出 d: tf.Tensor([2.], shape=(1,), dtype=float32)
b = tf.constant([11.0, 22.0], dtype=tf.float32, name='b')
print('b:',b)
#输出b: tf.Tensor([11. 22.], shape=(2,), dtype=float32)
c = tf.constant([[2.0, 3.0],[4.0,5.0]], dtype=tf.float32, name='c')
#输出print('c:',c)
#c: tf.Tensor(
#[[2. 3.]
# [4. 5.]], shape=(2, 2), dtype=float32)
