import tensorflow as tf
import matplotlib.pyplot as plt
# from __future__ import division, print_function
import numpy as np

# import MNIST Data
"""
1. 导入 tensorflow、matplotlib、random 和 numpy。然后，导入 mnist 数据集并进行独热编码
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

"""
2. 仔细观察一些数据有助于理解 MNIST 数据集。了解训练数据集中有多少张图片，测试数据集中有多少张图片。
   可视化一些数字，以便了解它们是如何表示的。
"""


def train_size(num):
    print('Total Training Images in Dataset =' + str(mnist.train.images.shape))
    print('------------------------------------------------------------------')
    x_train = mnist.train.images[:num, :]
    print('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num, :]
    print('y_train Examples Loaded = ' + str(y_train.shape))
    return x_train, y_train


# def test_size(num):
#     print('Total Test Images in Dataset =' + str(mnist.train.images.shape))
#     print('------------------------------------------------------------------')
#     x_test = mnist.test.images[:num, :]
#     print('x_train Examples Loaded = ' + str(x_test.shape))
#     y_test = mnist.test.labels[:num, :]
#     print('y_train Examples Loaded = ' + str(y_test.shape))
#     return x_test, y_test


def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])
    plt.title('Example: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))  # cmap常用于改变绘制风格，如黑白gray，翠绿色virdidis
    plt.show()


def display_mult_flat(start, stop):
    images = x_train[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))  # cmap常用于改变绘制风格，如黑白gray，翠绿色virdidis
    plt.show()


x_train, y_train = train_size(55000)
display_digit(np.random.randint(0, x_train.shape[0]))
display_mult_flat(0, 400)

# Parameters
learning_rate = 0.001
training_iters = 500
batch_size = 128
display_step = 10
# Network Parameters
n_input = 784
# Mnist data input
n_classes = 10
# Mnist total classes (0-9 digits)
dropout_eta = 0.85
# Dropout, probability to keep units

"""
4. 设置 TensorFlow 计算图的输入。定义两个占位符来存储预测值和真实标签
"""
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

"""
5. 定义一个输入为 x，权值为 W，偏置为 b，给定步幅的卷积层。激活函数是 ReLU，padding 设定为 SAME 模式：
"""


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


"""
6. 定义一个输入是 x 的 maxpool 层，卷积核为 ksize 并且 padding 为 SAME：
"""


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


"""
7. 定义 convnet，其构成是两个卷积层，然后是全连接层，一个 dropout 层，最后是输出层：
"""


def conv_net(x, weights, biases, dropout_eta):
    # reshape the input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # First convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling used for downsampling
    pool1 = maxpool2d(conv1, k=2)
    # Second convolution layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling used for downsampling
    pool2 = maxpool2d(conv2, k=2)
    resahpe1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # Fully connected layer
    fc1 = tf.add(tf.matmul(resahpe1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout_eta)
    # out
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


"""
8. 定义网络层的权重和偏置。第一个 conv 层有一个 5×5 的卷积核，1 个输入和 32 个输出。第二个 conv 层有一个 5×5 的卷积核，
32 个输入和 64 个输出。全连接层有 7×7×64 个输入和 1024 个输出，而第二层有 1024 个输入和 10 个输出对应于最后的数字数目。
所有的权重和偏置用 randon_normal 分布完成初始化：
"""
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
9. 建立一个给定权重和偏置的 convnet。定义基于 cross_entropy_with_logits 的损失函数，并使用 Adam 优化器进行损失最小化。优化后，计算精度
"""
pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

"""
10. 启动计算图并迭代 training_iterats次，其中每次输入 batch_size 个数据进行优化。
请注意，用从 mnist 数据集分离出的 mnist.train 数据进行训练。每进行 display_step 次迭代，会计算当前的精度。
最后，在 2048 个测试图片上计算精度，此时无 dropout。
"""
train_loss = []
train_acc = []
test_acc = []
with tf.compat.v1.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_eta})
        if step % display_step == 0:
            loss_train, acc_train = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                          y: batch_y,
                                                                          keep_prob: 1.})
            print('Iter:' + str(step) + ' , Minibatch Loss = ' + '{:.2f}'.format(loss_train)
                  + 'Accuracy = ' + '{:.2f}'.format(acc_train))
            # 计算测试集的准确率
            acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
            print('Testing Accuracy:' + '{:.2f}'.format(acc_test))
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)
        step += 1

"""
11. 画出每次迭代的 Softmax 损失以及训练和测试的精度：
"""
eval_indices = range(0, training_iters, display_step)
# 画loss
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Softmax Loss')
plt.show()

# 画训练和测试准确率
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
