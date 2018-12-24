
# -*- coding: utf-8 -*-)
import tensorflow as tf

with tf.name_scope("my_name_scope"):
    v1 = tf.get_variable(name = "v1", initializer=[1.0], dtype=tf.float32)
    v2 = tf.Variable(2.0, name="v2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  #输出 v1:0
print(v2.name)  #输出 my_name_scope/v2:0,my_scope是空间名
print(a.name)   #add操作张量名 my_name_scope/Add:0,my_name_scope是空间名
   
with tf.variable_scope("my_variable_scope"):
    v3 = tf.get_variable(name = "v3", initializer=[1.0], dtype=tf.float32)
    v4 = tf.Variable(2.0, name="v4", dtype=tf.float32)
    b = tf.add(v1, v2)
                
print(v3.name)  # my_variable_scope/v3:0
print(v4.name)  # my_variable_scope/v4:0
print(b.name)   # my_variable_scope/Add:0
                
