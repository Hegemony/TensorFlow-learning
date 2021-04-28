import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

"""
使用 TensorFlow 如 Contrib（层）来定义神经网络层及使用 TensorFlow 自带的优化器来计算和使用梯度。
Contrib 可以用来添加各种层到神经网络模型，如添加构建块。这里使用的一个方法是 tf.contrib.layers.fully_connected.
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
"""

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
batch_size = 10
n_hidden = 30
eta = 0.0001

"""
3. 定义模型
"""


def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, n_classes, activation_fn=None, scope='out')
    return out


"""
4. 为训练数据创建占位符,定义loss
"""

x = tf.placeholder(tf.float32, [None, n_input], name='placeholder_x')
y = tf.placeholder(tf.float32, [None, n_classes], name='placeholder_y')
y_hat = multilayer_perceptron(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y), name='loss')

train = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

"""
5. 定义计算精度 accuracy 的操作：
"""
acct_mat = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

"""
6. 初始化变量(该步骤在执行计算图前一行调用)
"""
init = tf.global_variables_initializer()

"""
7. 执行计算图
"""
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = int(mnist.train.num_examples / batch_size)
        for i in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train, loss], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c / batch_steps
        print('Epoch %02d, Loss = %.6f' % (epoch, epoch_loss))
    # 测试模型
    """
    eval和run都是获取当前结点的值的一种方式:
    两者的区别主要在于，eval一次只能得到一个结点的值，而run可以得到多个。
    """
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}), '%')
