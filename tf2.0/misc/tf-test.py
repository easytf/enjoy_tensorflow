import tensorflow as tf

v = tf.Variable(5)

@tf.function
def find_next_odd():
  v.assign(v + 1)
  if tf.equal(v % 2, 0):
    v.assign(v + 1)
  return v

print find_next_odd()
print v
