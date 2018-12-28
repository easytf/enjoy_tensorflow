#-*- coding:utf-8 -*-)
import tensorflow as tf

with tf.Session() as sess:
    d = tf.to_float(tf.reshape(tf.range(1,17),[4,4]))
    sess.run(tf.global_variables_initializer())
    print('shape:',sess.run(tf.shape(d)))
    print('src:',sess.run(d))
    
    # 矩阵有一半左右的元素变为element/0.5,其余为0
    dropout1 = tf.nn.dropout(d, 0.5, noise_shape = None)
    print(sess.run(dropout1))

    # 行大小相同4，行同为0，或同不为0
    dropout2 = tf.nn.dropout(d, 0.5, noise_shape = [4,1])
    print(sess.run(dropout2))
    
    # 列大小相同4，列同为0，或同不为0
    dropout3 = tf.nn.dropout(d, 0.5, noise_shape = [1,4])
    print(sess.run(dropout3))
    #不相等的noise_shape只能为1


