#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

a = tf.constant([1.0, 2.0], dtype=tf.float32, name = 'a')
b = tf.constant([3.0, 4.0], dtype=tf.float32, name = 'b')

summary_writer = tf.summary.create_file_writer('log', flush_millis=1000)
summary_writer.as_default()

with tf.GradientTape() as g:
    g.watch(a)
    c = a + b
    dy_dx = g.gradient(c, [a,b])
    print('a:',a)
    print('b:',b)
    print('c:',c)
    print('dy_dx:',dy_dx)
    with tf.summary.summary_scope("add"):
        tf.summary.write('c', a, 1)
        tf.summary.write('c', b, 2)
        tf.summary.write('c', c, 3)
        tf.summary.write('c', dy_dx, 4)
        tf.summary.flush()
