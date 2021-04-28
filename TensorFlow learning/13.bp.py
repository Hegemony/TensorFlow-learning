import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
1. 加载数据集，使用one-hot编码
"""
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

"""
2. 定义超参数和其他常量。这里，每个手写数字的尺寸是 28×28=784 像素。数据集被分为 10 类，以 0 到 9 之间的数字表示。
   这两点是固定的。学习率、最大迭代周期数、每次批量训练的批量大小以及隐藏层中的神经元数量都是超参数。
   可以通过调整这些超参数，看看它们是如何影响网络表现的：
"""
n_input = 784
n_classes = 10

# 超参数
max_epochs = 10000
learning_rate = 0.5
batch_size = 10
seed = 0
n_hidden = 30

"""
3. 需要 Sigmoid 函数的导数来进行权重更新，所以定义它：
"""


def sigmaprime(x):
    """
    tf.multiply(x,y,name=None) 乘法，相同位置的元素相乘
    tf.subtract(x,y,name=None) 张量x减去张量y（元素级减法操作）
    :param x:
    :return:
    """
    return tf.multiply(tf.sigmoid(x), tf.subtract(tf.constant(1.0), tf.sigmoid(x)))


"""
4. 为训练数据创建占位符：
"""
x_in = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

"""
5. 创建模型：
"""


def multilayer_perceptron(x, weights, biases):
    # hidden layer with Sigmoid actication
    h_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    out_layer_1 = tf.sigmoid(h_layer_1)
    # output layer with linear activation
    h_out = tf.matmul(out_layer_1, weights['out']) + biases['out']
    return tf.sigmoid(h_out), h_out, out_layer_1, h_layer_1


"""
6. 定义权重和偏置变量：
"""
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden], seed=seed)),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], seed=seed))
}

biases = {
    'h1': tf.Variable(tf.random_normal([1, n_hidden], seed=seed)),
    'out': tf.Variable(tf.random_normal([1, n_classes], seed=seed))
}

"""
7. 为正向传播、误差、梯度和更新计算创建计算图：
"""
# forward
y_hat, h_2, o_1, h_1 = multilayer_perceptron(x_in, weights, biases)

# error
err = y_hat - y

# backward
delta_2 = tf.multiply(err, sigmaprime(h_2))
delta_w_2 = tf.matmul(tf.transpose(o_1), delta_2)

wtd_error = tf.matmul(delta_2, tf.transpose(weights['out']))
delta_1 = tf.multiply(wtd_error, sigmaprime(h_1))
delta_w_1 = tf.matmul(tf.transpose(x_in), delta_1)

eta = tf.constant(learning_rate)

# update weights
step = [
    tf.assign(weights['h1'], tf.subtract(weights['h1'], tf.multiply(eta, delta_w_1))),
    tf.assign(biases['h1'], tf.subtract(biases['h1'], tf.multiply(eta, tf.reduce_mean(delta_1, axis=[0])))),
    tf.assign(weights['out'], tf.subtract(weights['out'], tf.multiply(eta, delta_w_2))),
    tf.assign(biases['out'], tf.subtract(biases['out'], tf.multiply(eta, tf.reduce_mean(delta_2, axis=[0])))),
]

"""
8. 定义计算精度 accuracy 的操作：
"""
acct_mat = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

"""
9.初始化变量
"""
init = tf.global_variables_initializer()

"""
10. 执行图
"""
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(max_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(step, feed_dict={x_in: batch_xs, y: batch_ys})
        if epoch % 1000 == 0:
            acc_test = sess.run(accuracy, feed_dict={x_in: mnist.test.images, y: mnist.test.labels})
            acc_train = sess.run(accuracy, feed_dict={x_in: mnist.train.images, y: mnist.train.labels})
            print('Epoch:{0} Accuracy Train:{1}% Accuracy Test:{2}%'.format(epoch, acc_train / 600, (acc_test / 100)))
