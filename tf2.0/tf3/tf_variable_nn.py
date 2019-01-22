# -*- coding: utf-8 -*-)
import tensorflow as tf

#定义变量空间,并且设置reuse标志为AUTO_RESUSE
with tf.name_scope("get_variable"):
    #使用随机函数生成2×3维的张量，标准差是1,随机种子是1
    w1 = tf.Variable(initial_value = tf.random.normal([2,3], stddev = 1, seed = 1),  name = 'w1')
    #创建第二个变量，我们故意使用上一个变量的值
    zeros = tf.zeros_initializer()
    w2 = tf.Variable(initial_value = zeros([2,3]),  name = 'w1')
    #创建第三个变量w3时，tensorflow发现w3不存在，便用initial_value初始化器创建新的变量
    ones = tf.ones_initializer()
    w3 = tf.Variable(initial_value = ones([2,3]),  name = 'w3')
bias = tf.Variable(initial_value = tf.constant([[0.1, 0.2,0.3],[0.4,0.5,0.6]]))
x = tf.constant([[0.3,0.5],[0.1,0.2]])

y1 = tf.matmul(x, w1) + bias
y2 = tf.matmul(x, w2) + bias

print('y1 =', y1)
print('y2 =', y2)
print('w3 =', w3)
