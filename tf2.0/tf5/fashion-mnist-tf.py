# coding: utf-8
#导入Fashion MNIST数据集 
# 60,000图片用作训练集, 10,000张图片用作测试集.
import numpy as np
import tensorflow as tf
from tensorflow import keras
#如果是远程ssh运行代码，请打开下面两行
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
#从~/.keras/datasets/fashion-mnist/加载数据到numpy数组
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#mnist是从0-9的10个数字，fashion是10种不同类型的服饰，也有包包,每个物品在数组中索引也就是
#它在模型中的标签，这样和mnist的标签完全一致;
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images[0])
#打印下训练图片的维度 60,000 images 28 x 28 = (60000, 28, 28)
print('train_images.shape:',train_images.shape)
#打印tensorflow版本
print(tf.__version__)
#看下标签长啥样的 
print('labels:',train_labels)

#同样查看下测试数据信息
print('test_images.shape:',test_images.shape)
print('len(test_labels):',len(test_labels))

#每张图片大小是28×28，读到numpy的数组也是28×28的二维数组,每个像素值0-255
#显示第一张图片
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#对每个像素进行归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

#显示前面25张图片和对应的标签
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#搭建神经网络模型
#这段就是符号定义模型
model = keras.Sequential([
    #把28×28的数组打平成一维数组
    keras.layers.Flatten(input_shape=(28, 28)),
    #添加一层全连接神经网络层，有256个神经元结点
    #激活函数是relu
    keras.layers.Dense(256, activation='relu'),
    #再添加一层，有128个神经元结点
    keras.layers.Dense(128, activation='relu'),
    #最后一层，有10个神经元结点
    #激活函数用的是softmax,还记得我们前面讲过交叉熵损失函数吗？
    keras.layers.Dense(10, activation='softmax')
])

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
#@tf.function
def train_step(image, label):
  with tf.GradientTape() as tape:
    image = tf.reshape(image, (-1, 28*28))
    predictions = model(image)
    print('predictions:',predictions)
    print('label:',label)
    loss = loss_object(label, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  print('loss:', loss)
  train_accuracy(label, predictions)

#验证模型
@tf.function
def test_step(image, label):
  predictions = model(image)
  t_loss = loss_object(label, predictions)
  
  test_loss(t_loss)
  test_accuracy(label, predictions)

#训练模型次数
EPOCHS = 5

for epoch in range(EPOCHS):
  for image, label in zip(train_images,train_labels):
    train_step(image, label)
  
  for test_image, test_label in zip(train_test,test_labels):
    test_step(test_image, test_label)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))


