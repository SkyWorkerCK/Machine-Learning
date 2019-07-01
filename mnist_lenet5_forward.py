# coding: utf-8
import tensorflow as tf
IMAGE_SIZE = 28         # 图片分辨率大小28*28
NUM_CHANNELS = 1        # 单通道灰度图，3通道是RGB
CONV1_SIZE = 5          # 第一层卷积核的大小为5*5
CONV1_KERNEL_NUM = 32   # 第一层卷积核的数量
CONV2_SIZE = 5          # 第二层卷积核的大小为5*5
CONV2_KERNEL_NUM = 64   # 第二层卷积核的数量
FC_SIZE = 512           # 第一层全连接神经网络的隐藏层神经元个数是512个
OUTPUT_NODE = 10        # 第二层全连接神经网络的隐藏层神经元个数是10个

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

# 卷积操作
def conv2d(x, w):
    # tf.nn.conv2d([batch, 行分辨率， 列分辨率，通道数],
    #               [卷积核行分辨率，卷积核列分辨率，通道数，核个数],
    #               [1，行滑动步长，列滑动步长，1]
    #               [padding = 'SAME])
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化操作
def max_pool_2x2(x):
    # tf.nn.max_pool([batch, 行分辨率， 列分辨率，通道数],
    #               [1，池化核大小，池化核大小，1]，
    #               [1，池化核滑动步长，池化核滑动步长，1]，
    #               padding = 'SAME')
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x, train, regularizer):
    # 初始化第一层卷积核
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 初始化第一层偏置值
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 接下来是卷积操作
    conv1 = conv2d(x, conv1_w)
    # 对卷积后的操作进行添加偏置,并经过激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 对于激活后的结果进行最大池化操作
    pool1 = max_pool_2x2(relu1)


    # 初始化第二层卷积核
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    # 初始化第二层偏置值
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 接下来是卷积操作
    conv2 = conv2d(pool1, conv2_w)
    # 对卷积后的操作进行添加偏置,并经过激活函数
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 对于激活后的结果进行最大池化操作
    pool2 = max_pool_2x2(relu2)


    # 将pool2的维度保存在pool_shape中
    pool_shape = pool2.get_shape().as_list()
    # 求出所有特征点，将多维变成两维
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 将pool2表示成[pool_shape[0], nodes]的二位形式
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])


    # 将reshaped喂入第一层全连接神经网络
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是在训练阶段，则采用概率为50%的dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 喂入第二层全连接神经网络
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y



