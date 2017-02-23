# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:14:02 2017

@author: shangguanxf
"""
#先运行下面文件得到input_data包
runfile('D:/Program Files/Anaconda3/Lib/site-packages/tensorflow/examples/tutorials/mnist/fully_connected_feed.py', wdir='D:/Program Files/Anaconda3/Lib/site-packages/tensorflow/examples/tutorials/mnist')
import input_data
#得到数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
#通过占位符预先定义x、W、b的大小和类型
x=tf.placeholder('float',[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#定义函数，得到预测分类
y = tf.nn.softmax(tf.matmul(x,W) + b)
#实际分类
y_ = tf.placeholder("float", [None,10])
#定义交叉熵作为loss函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#选用梯度下降算法对神经网络进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化变量的操作
init = tf.global_variables_initializer()
#创建Session，初始化变量
sess = tf.Session()
sess.run(init)
#为x、y_赋值并训练，每次训练只取部分数据，即sgd
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#根据比对判断预测结果是否正确。argmax得到最大值的索引，argmax(,0)按行，argmax(,1)按列
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#得到正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#运行并输出结果
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))