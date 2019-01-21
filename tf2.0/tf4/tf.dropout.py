#-*- coding:utf-8 -*-)
import tensorflow as tf

d = tf.cast(tf.reshape(tf.range(1,17),[4,4]), dtype=tf.float32)
print('src:',d)

# 输出keep_prob比例的元素，输出元素的值element/keep_prob,其余为0
dropout1 = tf.nn.dropout(d, 0.2, noise_shape = None)
print('only keep_prob:\n',dropout1)
# 行大小相同4，行同为0，或同不为0
dropout2 = tf.nn.dropout(d, 0.4, noise_shape = [4,1])
print('keep_prob and [4,1] noise:\n', dropout2)
# 列大小相同4，列同为0，或同不为0
dropout3 = tf.nn.dropout(d, 0.4, noise_shape = [1,4])
print('keep_prob and [1,4] noise:\n',dropout3)

