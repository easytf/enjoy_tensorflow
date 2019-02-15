# -*- coding: utf-8 -*-
import tensorflow as tf

a = [1.0,2.0,3.0,4.0]
b = [[0.3, 0.1], [0.0, 0.2]]

sa = tf.nn.softmax(a)
sb =  tf.nn.softmax(b)
sb1 =  tf.nn.softmax(logits = b, axis = 0)
print( sa.numpy() )
print( sb.numpy() )
print( sb1.numpy() )
