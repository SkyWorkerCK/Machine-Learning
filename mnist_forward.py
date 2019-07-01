import tensorflow as tf
import tensorflow.contrib as contrib

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    # tf.truncated_normal()从截断的正态分布中输出随机值, shape表示生成张量的维度，stddev是标准差
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 将每个变量的正则化损失加入集合losses中
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    # tf.nn.relu()为激活函数，当x<0时，y=0; 当x>=0时，y=x
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y


