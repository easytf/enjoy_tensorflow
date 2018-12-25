
import tensorflow as tf
import numpy as np
import random

def product(a, b): 
    x = tf.placeholder(tf.float32, shape=[1])
    y = tf.placeholder(tf.float32, shape=[1])
    h = 0.01
    z = ( x + random.random()*h )*( y + random.random()*h )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(z, feed_dict={x:a, y:b})
        return(c)

a = np.array([3]) 
b = np.array([4]) 
print(product(a, b))

def product2(a, b): 
    x = tf.placeholder(tf.float32, shape=[1])
    y = tf.placeholder(tf.float32, shape=[1])
    h = 0.01
    z = ( x + h*y )*( y + h*x )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(z, feed_dict={x:a, y:b})
        return(c)

a = np.array([-3]) 
b = np.array([-4]) 
print(product2(a, b))
