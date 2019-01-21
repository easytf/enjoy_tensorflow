#-*- coding:utf-8 -*-)
import tensorflow as tf
import numpy as np

#初始化
X = tf.constant([[2,4,-2]], dtype=tf.float32)
W = tf.Variable(tf.random.normal(shape=[3,2], stddev =1, seed = 1))
#标签label
ycap = [0]
#number of training of examples
num_examples = 1
#设置步长
h = 0.01
#前向传播函数
y = tf.matmul(X, W)

y_o = tf.sigmoid( y )
#计算Loss
loss = -tf.reduce_sum( tf.math.log( y_o.numpy()[ range(num_examples) ,ycap ] ) )
print('loss:',loss.numpy())     #1.7505805 每次结果可能不一样，因为每次初始化权重是随机数

#反向传播
def active_sigmoid( x):
    return tf.multiply(tf.sigmoid(x), (1 - tf.sigmoid(x))) 

y_oo = np.copy(y_o.numpy())
#实现损失函数对y_o的微分
y_oo[ range(num_examples), ycap ] = 1/-( y_oo[range(num_examples), ycap] )
temp = np.zeros_like(y_o)

temp[range(num_examples), ycap] = 1
dcost = tf.multiply(temp, y_oo)
#y_o对y的微分
dy_o = active_sigmoid(y)
#元素相乘
dgrad = tf.multiply(dcost, dy_o)
#对X进行矩阵转置
XT = tf.transpose(X)
dw = tf.matmul(XT, dgrad)
#更新权重
W = W - h * dw
#再次使用更新的权重来计算loss
y = tf.matmul(X,W)

yo = tf.sigmoid(y) 
loss = -tf.reduce_sum(tf.math.log(yo.numpy()[range(num_examples),ycap]))
print('update loss:',loss.numpy())     #1.5896498 每次结果可能不一样，因为每次初始化权重是随机数

