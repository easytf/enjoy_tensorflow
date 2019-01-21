# -*- coding: utf-8 -*-
import tensorflow as tf

tf.enable_eager_execution()
a = [1.0,2.0,3.0,4.0]
b = [0.3,0.1,0.0]
print(tf.nn.softmax(a))
print(tf.nn.softmax(b))
