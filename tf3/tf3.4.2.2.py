
#-*- coding:utf-8 -*-)
import tensorflow as tf
import numpy as np

#initialization
X = tf.constant([[2,4,-2]], dtype=tf.float32)
print(X)

W = tf.Variable(tf.random_normal(shape=[3,2], stddev =1, seed = 1))
#label
ycap = [0]
#number of training of examples
num_examples = 1
#step size
h =0.01
#forward-propogation
y = tf.matmul(X, W)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #loss calculation
    y_o = sess.run(tf.sigmoid( y ) )
    loss = -tf.reduce_sum( tf.log( y_o[ range(num_examples) ,ycap ] ) )
    print('loss:',sess.run(loss))     #outputs 7.87 (for you it would be different due to random initialization of weights.)

#backprop starts
def derivative_sigmoid( x):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run( tf.multiply(tf.sigmoid(x), (1 - tf.sigmoid(x))) )

temp1 = np.copy(y_o)
#implementation of derivative of cost function with respect to y_o
temp1[ range(num_examples) ,ycap ] = 1/-( temp1[range(num_examples),ycap] )
temp = np.zeros_like(y_o)

temp[range(num_examples),ycap] = 1
print('temp:',temp)
print('temp1:',temp1)
#derivative of cost with respect to y_o
dcost = tf.multiply(temp,temp1)
#derivative of y_o with respect to y
dy_o = derivative_sigmoid(y)
#element-wise multiplication
dgrad = tf.multiply(dcost, dy_o)
XT = tf.transpose(X)
dw = tf.matmul(XT, dgrad)
#weight-update
W = W - h * dw
#forward prop again with updated weight to find new loss
y = tf.matmul(X,W)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    yo = sess.run( tf.sigmoid(y) )
    loss = -tf.reduce_sum(tf.log(yo[range(num_examples),ycap]))
    print('update loss:',sess.run(loss) )     #outpus 7.63 (again for you it would be different!)

