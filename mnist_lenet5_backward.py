import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005     # 初始学习率
LEARNING_RATE_DECAY = 0.99   # 学习衰减率
REGULARIZER = 0.0001         # 正则率
STEPS = 50000                # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"# 模型保存路径
MODEL_NAME = "mnist_model"  # 模型名称

def backward(mnist):
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,                             # 一次喂入神经网络的数量
        mnist_lenet5_forward.IMAGE_SIZE,        # 行分辨率
        mnist_lenet5_forward.IMAGE_SIZE,        # 列分辨率
        mnist_lenet5_forward.NUM_CHANNELS      # 输入的通道数
    ])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    # 正向传输推算出模型
    y = mnist_lenet5_forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 下面算出包括正则化的损失函数
    # logits为神经网络输出层的输出
    # 传入的label为一个一维的vector，长度等于batch_size，每一个值的取值区间必须是[0，num_classes)，其实每一个值就是代表了batch中对应样本的类别
    # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 矩阵中所有元素求平均值
    cem = tf.reduce_mean(ce)
    # tf.get_collection(‘losses’)：返回名称为losses的列表
    # tf.add_n(list)：将列表元素相加并返回
    loss = cem + tf.add_n(tf.get_collection('losses'))



    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )


    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    # 定义滑动平均
    # tf.train.ExponentialMovingAverage()这个函数用于更新参数，就是采用滑动平均的方法更新参数
    # MOVING_AVERAGE_DECAY是衰减率,用于控制模型的更新速度,设置为接近1的值比较合理
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # apply()方法添加了训练变量的影子副本
    # 返回值：ExponentialMovingAverage对象，通过对象调用apply方法可以通过滑动平均模型来更新参数。
    # tf.trainable_variables返回的是需要训练的变量列表
    # tf.all_variables返回的是所有变量的列表
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        # 里面的内容需要在将train_step、ema_op执行完后才能执行
        # tf.no_op()表示执行完 train_step, ema_op 操作之后什么都不做
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 通过设置ckpt实现断点续训的功能
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 对数据集中的xs进行reshape操作
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.NUM_CHANNELS
            ))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s) , loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()
    # BATCH_SIZE = 200
    # mnist = input_data.read_data_sets("../MNIST-data", one_hot=True)
    # xs, ys = mnist.train.next_batch(BATCH_SIZE)
    # print("xs=", xs.shape)
    # print("ys=", ys.shape)




