# coding: utf-8
#导入Fashion MNIST数据集 
# 60,000图片用作训练集, 10,000张图片用作测试集.
import numpy as np
import tensorflow as tf
from tensorflow import keras
#如果是远程ssh运行代码，请打开下面两行
#import matplotlib
#matplotlib.use('Agg')
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

#optimizer:使用adam梯度下降优化器
#loss:使用sparse损失函数
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#开始训练模型
model.fit(train_images, train_labels, epochs=10)
#用测试集验证模型的loss和accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

#对测试数据集进行预测
predictions = model.predict(test_images)

# 打印第一个预测的结果数组
print('predictions first:',predictions[0])
#取数组中概率最大的一个
print('predict label:', tf.argmax(predictions[0]))
#看下和实际标签是什么
print('test label:', test_labels[0])
#把实际的标签和推理的标签做对比，如果不一样，则显示红色，如果相同,则显示蓝色
def plot_image(i, predictions_array, true_label, img):
  #设置显示方式
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  #关闭背景的网格线
  plt.grid(False)
  #关闭x轴，y轴坐标刻度
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  #predictions_array中概率值最大的即为标签
  predicted_label = tf.argmax(predictions_array)
  #判断预测标签和实际标签是否相等
  if predicted_label == true_label:
    #预测和实际相符就显示蓝色
    color = 'blue'
  else:
    #否则就显示红色
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  #设置y轴的范围
  plt.ylim([0, 1]) 
  predicted_label = tf.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#打印第一张图片看看效果如何 
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

#打印15张图片，每个单元由测试图片，预测标签，实际标签组成
#5行 3列
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

