# -*- coding: utf-8 -*-)
import tensorflow as tf

#定义变量空间,并且设置reuse标志为AUTO_RESUSE
with tf.variable_scope("get_variable", reuse=tf.AUTO_REUSE):
    #使用随机函数生成2×3维的张量，标准差是1,随机种子是1
    w1 = tf.get_variable(initializer = tf.random_normal([2,3], stddev = 1, seed = 1),  name = 'w1')
    #创建第二个变量，我们故意使用上一个变量的值
    w2 = tf.get_variable(initializer = tf.zeros([2,3]),  name = 'w1')
    #创建第三个变量w3时，tensorflow发现w3不存在，便用initializer初始化器创建新的变量
    w3 = tf.get_variable(initializer = tf.ones([2,3]),  name = 'w3')
bias = tf.Variable(initial_value = tf.constant([[0.1, 0.2,0.3],[0.4,0.5,0.6]]))
x = tf.constant([[0.3,0.5],[0.1,0.2]])

y1 = tf.matmul(x, w1) + bias
y2 = tf.matmul(x, w2) + bias

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('y1 =', sess.run(y1))
    print('y2 =', sess.run(y2))
    print('w3 =', sess.run(w3))
