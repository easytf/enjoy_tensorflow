import tensorflow as tf
import numpy as np

def product(x, y): 
    h = 0.01
    z = tf.multiply(( x + tf.random.normal(shape=[1])*h ), ( y + tf.random.normal(shape=[1])*h ))
    return z

a = np.array([3]) 
b = np.array([4]) 
c = product(a, b)
print( c.numpy())

def product2(x, y): 
    h = 0.01
    z = tf.multiply(( x + h*y ),( y + h*x ))
    return z

a = np.array([-3]) 
b = np.array([-4]) 
c = product2(a, b)
print( c.numpy())
