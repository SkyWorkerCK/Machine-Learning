import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

# 时间间隔是1s
TEST_INTERVAL_SECS = 1

def test(mnist):
    # 定义一张图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        # 通过使用tf.variables_to_restore()函数，
        # 可以使在加载模型的时候将影子变量直接映射到变量的本身，
        # 所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值
        # 而不需要去获取影子变量。
        # 我们也知道，其实在TensorFlow中，变量的滑动平均值都是由影子变量所维护的，如果你想要获取变量的滑动平均值需要获取的是影子变量而不是变量本身
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()