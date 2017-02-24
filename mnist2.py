# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:44:14 2017

@author: shangguanxf
"""
#导入数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#使用InteractiveSession.入门中使用Session,需要在启动Session前构建好整个计算图。故在最后才定义Session
import tensorflow as tf
sess=tf.InteractiveSession();
#通过占位符给x，y_占位
x=tf.placeholder('float',shape=[None,784]);
y_=tf.placeholder('float',shape=[None,10]);
#定义变量
W=tf.Variable(tf.zeros([784,10]));
b=tf.Variable(tf.zeros([10]));
y= tf.nn.softmax(tf.matmul(x,W) + b)
#初始化变量
sess.run(tf.global_variables_initializer())
#用交叉熵为损失函数
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#使用最速下降法作寻优
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#训练模型
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#模型评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#mnist进阶
#创建初始化权重函数
#truncated_normal截断正态分布
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义卷积层和池化层
#padding='VALID' 不移动出边缘，导致结果变小
#padding='SAME' 可以移出边缘，边缘外部分补0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#shape前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。

#第一层的权值向量和偏置项
#感觉为输入xa+b,a为32维向量，输出变为32维。即32个神经元
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#将x转换为28*28的图片格式。-1表示自动计算这一维的大小。第一位为数目，二三位为图片宽高，最后一位为图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1,28,28,1])
#用relu函数作为激活函数，对第一层卷积进行池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层的权值向量和偏置项
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#用relu激活第二层的隐层，并池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))