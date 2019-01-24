import tensorflow as tf
import numpy as np

def product(x, y): 
    return tf.multiply(x, y)

def add(x, y): 
    return tf.add(x, y) 

def forward(a, b, c):
    d = tf.add(a, b)
    return product(d, c)

a =np.array([5])
b =np.array([-3])
c =np.array([-2])
print(forward(a, b, c))

d = tf.add(a, b)
h = 0.01
df_wrt_d = tf.cast(c, tf.float32)
df_wrt_c = tf.cast(d, tf.float32)
dd_wrt_a = 1.0
dd_wrt_b = 1.0

df_wrt_a = tf.multiply(df_wrt_d,dd_wrt_a)
df_wrt_b = tf.multiply(df_wrt_d,dd_wrt_b)

a = a + h*df_wrt_a
b = b + h*df_wrt_b
c = c + h*df_wrt_c
def bakcforward(a, b, c):
    d = tf.add(a, b)
    return product(d, c)

print(bakcforward(a, b,c ))

