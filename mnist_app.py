import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
    # 复现计算图
    with tf.Graph().as_default() as tg:
        # 搭建整个计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        # 实例化带有滑动平均的saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 恢复ckpt
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            # 如果ckpt存在，则将ckpt恢复到当前会话
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 将准备好的图片喂入神经网络
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

def pre_pic(picName):
    # 打开图片
    img = Image.open(picName)
    # 将图片重新规格为28*28像素的图片（采用锯齿的方式）
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 采用reIm.convert('L')将图片变成灰度图
    # np.array()将灰度图变成矩阵的形式
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 遍历每个像素点进行反色，将每个像素点变成只有白色或者黑色,白色为255
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0                # 表示纯黑色点
            else:
                im_arr[i][j] = 255              # 表示纯白色点

    # 将反色后的图片变成一个行向量
    nm_arr = im_arr.reshape([1, 784])
    # 将向量中所有的元素变成浮点型
    nm_arr = nm_arr.astype(np.float32)
    # 将向量中所有的元素的值从0-255转变成0-1之间的值
    # 矩阵点乘则要求参与运算的矩阵必须是相同维数的，是每个对应元素的逐个相乘。
    # np.multiply()表示矩阵点乘，tf.matmul()为矩阵乘法
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def application():
    testnum = input("Please input the number of test pictures:")
    testnum = int(testnum)
    for i in range(testnum):
        testpic = input("Please input the location of the picture:")
        # 对图片进行预处理
        testPicAcc = pre_pic(testpic)
        preValue = restore_model(testPicAcc)
        print("The prediction number is:", preValue)


def main():
    application()

if __name__ == '__main__':
    main()