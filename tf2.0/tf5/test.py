import tensorflow as tf

X = tf.random.normal(shape=[2,1], mean=2, stddev=2)
Y = [[0.1]]
y_true = tf.matmul(X,Y) +0.2
print('X:',X.numpy())
print('Y:',Y)
print('tf.matmul(X,Y):',tf.matmul(X,Y).numpy())
print('y_true:',y_true.numpy())
