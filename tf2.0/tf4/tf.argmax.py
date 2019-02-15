import tensorflow as tf

a = tf.Variable([[[1, 2, 3], [2, 3, 4]], [[5, 4, 3], [8, 7, 2]]])
print(a,'-->(a,0)')
print(tf.argmax(a, 0))

print(a,'-->(a,1)')
print(tf.argmax(a, 1))
 
b = tf.Variable([ [[1, 2, 3], [2, 3, 4]], [[5, 4, 3], [8, 7, 2]],[[9,1,3],[7,6,2]] ])
print(b,'-->(b,0)')
print(tf.argmax(b, 0))

print(b,'-->(b,1)')
print(tf.argmax(b, 1))

print(b,'-->(b,2)')
print(tf.argmax(b, 2))

c = tf.Variable([ [[1, 2], [3, 4]], [[4, 3], [7, 2]],[[9,3],[6,2]] ])
print(tf.argmax(c, 0))
print(tf.argmax(c, 1))
print(tf.argmax(c, 2))
 
b = np.array([ [[1, 2, 3], [2, 3, 4]], [[5, 4, 3], [8, 7, 2]],[[9,1,3],[7,6,2]] ])
print(tf.argmax(b, 0))
