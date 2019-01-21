#!/usr/bin/python3
# -*- coding: utf-8 -*-)

import tensorflow as tf

a = tf.constant([1.0, 2.0], dtype=tf.float32, name = 'a')
b = tf.constant([3.0, 4.0], dtype=tf.float32, name = 'b')
c = a + b

summary_writer = tf.summary.create_file_writer('./add', flush_millis=1000)
with summary_writer.as_default():
    with tf.summary.summary_scope("add"):
        tf.summary.write('c', c, 1)

#summary_op = tf.summary.merge_all()
#summary_writer = tf.summary.SummaryWriter('./log/')
#summary_writer.add_summary(summary, 0)

with tf.GradientTape() as g:
    dy_dx = g.gradient(y, [b])
    print('a:',a)
    print('b:',b)
    print('c:',c)
    print('dy_dx:',dy_dx)
