import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-1.,1,500)
target = tf.constant(0.)

#计算L2 loss
l2_y = tf.square(target - x)

#计算L1 loss
l1_y = tf.abs(target - x)

#用画图来体现损失函数的特点
plt.plot(x.numpy(), l1_y.numpy(), color = 'b', label = 'L1_loss')
plt.plot(x.numpy(), l2_y.numpy(), color = 'r', label = 'L2_loss')
plt.legend(loc='best')
plt.show()
