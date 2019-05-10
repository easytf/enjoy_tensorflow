#!/usr/bin/env python
# coding: utf-8

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#使用tensorflow dataset加载数据
dataset, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True)
#查看训练数据和标签信息
print(info.splits['train'].num_examples)
print(info.features['label'].num_classes)

mnist_train, mnist_test = dataset['train'], dataset['test']
#查看测试数据信息
print("type(mnist_train):",mnist_train)

#对灰度图中的每个像素进行归一化
def convert_types(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

#map调用convert_types函数归一化像素
mnist_train = mnist_train.map(convert_types)
#随机打乱数据
mnist_train = mnist_train.shuffle(10000)
#将数据集中组成32个元素一批
mnist_train = mnist_train.batch(32)
#所以我们也可以写成连锁的形式
#mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
#处理测试数据
mnist_test = mnist_test.map(convert_types).batch(32)

#构建keras模型，模型子类化
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    #加入卷积层提取特征
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(256, activation='relu')
    self.d2 = Dense(128, activation='relu')
    self.d3 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)
  
model = MyModel()

# Choose an optimizer and loss function for training:
#使用sparse损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#使用Adam优化器
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over epochs and then print the overall result.

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#使用梯度带实现自动微分
@tf.function
def train_step(image, label):
  with tf.GradientTape() as tape:
    predictions = model(image)
    loss = loss_object(label, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(label, predictions)

#验证模型
@tf.function
def test_step(image, label):
  predictions = model(image)
  t_loss = loss_object(label, predictions)
  
  test_loss(t_loss)
  test_accuracy(label, predictions)

#训练模型次数
EPOCHS = 500

for epoch in range(EPOCHS):
  for image, label in mnist_train:
    train_step(image, label)
  
  for test_image, test_label in mnist_test:
    test_step(test_image, test_label)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

# The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/alpha/tutorials/keras).
