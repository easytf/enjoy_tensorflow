import tensorflow as tf  

# calculate cross_entropy 
def my_cross(y, y_):
    #把模型向量转为概率
    ys = tf.nn.softmax(y)
    corss_entropy = -tf.reduce_sum(y_*tf.math.log(ys))
    
    return corss_entropy

#初步训练后模型最后一层的向量
y  = tf.Variable([1.5,3.0,2.5])
#真实值，实际的概率
y_ = tf.constant([0.0, 1.0, 0.0])  
#调用我们自己实现的交叉熵函数
cross_entropy = my_cross(y, y_)
print("preliminary cross_entropy = ", cross_entropy.numpy())  

#几经训练后模型最后一层的向量
y  = tf.Variable([1.5,6.0,2.0])
cross_entropy = my_cross(y, y_)
print("final cross_entropy = ", cross_entropy.numpy())  

#一步到位，直接使用TensorFlow的交叉熵库函数
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
print("softmax cross_entropy = ",cross_entropy.numpy())  

y_s = tf.constant(1)
cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = y_s))
print("sparse cross_entropy = ",cross_entropy.numpy())  

y  = tf.constant([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 4.0, 3.0],[1.0, 4.0, 3.0, 2.0]])
y_ = tf.constant([3, 2,1])
cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y, 1)))
cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = y_))
print("sparse cross_entropy = ",cross_entropy)  
print(tf.argmax(y, 1))

#####for tf.nn.sigmoid_cross_entropy_with_logits
#初步训练后模型最后一层的向量
y  = tf.Variable([1.5,3.0,2.5])
#真实值，实际的概率
y_ = tf.constant([0.0, 1.0, 1.0])  
cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = y, labels = y_))
print("sigmoid cross_entropy = ",cross_entropy.numpy())  


