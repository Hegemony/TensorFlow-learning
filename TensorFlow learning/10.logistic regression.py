import tensorflow as tf
import matplotlib.pyplot as plt

"""
1. 可以从模块 input_data 给出的 TensorFlow 示例中获取 MNIST 的输入数据。该 one_hot 标志设置为真，以使用标签的 one_hot 编码。
这产生了两个张量，大小为 [55000，784] 的 mnist.train.images 和大小为 [55000，10] 的 mnist.train.labels。
mnist.train.images 的每项都是一个范围介于 0 到 1 的像素强度
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
"""
2. 在 TensorFlow 图中为训练数据集的输入 x 和标签 y 创建占位符
"""
X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')
"""
3. 创建学习变量、权重和偏置
"""
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

"""
4.创建逻辑回归模型。TensorFlow OP 给出了 name_scope（"wx_b"）:
"""
with tf.name_scope('wx_b') as scope:
    y_hat = tf.nn.softmax(tf.matmul(X, W) + b)

"""
5. 训练时添加 summary 操作来收集数据。使用直方图以便看到权重和偏置随时间相对于彼此值的变化关系。
可以通过 TensorBoard Histogtam 选项卡看到:
"""
w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

"""
6. 定义交叉熵（cross-entropy）和损失（loss）函数，并添加 name scope 和 summary 以实现更好的可视化。
使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见：
"""
with tf.name_scope('cross-entropy') as scope:
    """
    tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
    除去name参数用以指定该操作的name，与方法有关的一共两个参数 ：
    第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]
                    num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率。
    第二个参数labels：实际的标签，大小同上。
    
    注意！！！这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,就是对向量里面所有元素求和，
    最后才得到，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_hat))
    tf.summary.scalar('cross-entropy', loss)

"""
7. 采用 TensorFlow GradientDescentOptimizer，学习率为 0.01。为了更好地可视化，定义一个 name_scope
"""
with tf.name_scope('Train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

"""
8. 为变量进行初始化：
"""
init = tf.global_variables_initializer()

"""
8-1. 计算准确率（）
"""
# tf.argmax(y, 1)返回1的位置(真实值)，tf.argmax(y_hat, 1)(预测值)返回概率值最大的位置，比较位置是否相等，若相等返回true，不等返回false，存放在布尔列表中
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_hat, 1))  # tf.argmax()返回一维张量中最大值的"位置"
# 求准确率
"""
tf.cast(x, dtype, name=None)
第一个参数 x:   待转换的数据（张量）
第二个参数 dtype： 目标数据类型
第三个参数 name： 可选参数，定义操作的名称
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
9. 组合所有的summary操作
"""
merged_summary_op = tf.summary.merge_all()

"""
10. 定义会话将所有的summary存储在定义的文件夹中：
"""
batch_size = 10
max_epochs = 100
with tf.compat.v1.Session() as sess:
    sess.run(init)
    # summary_writer = tf.summary.FileWriter('graphs', sess.graph)
    # Training
    for epoch in range(max_epochs):
        loss_avg = 0
        num_of_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_of_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)  # 得到下一个batch
            _, l, summary_str = sess.run([optimizer, loss, merged_summary_op], feed_dict={X: batch_xs, Y: batch_ys})
            loss_avg += l
            # summary_writer.add_summary(summary_str, epoch * num_of_batch + i)
        loss_avg = loss_avg / num_of_batch
        print('Epoch {0} : Loss {1}'.format(epoch, loss_avg))
        print('Done')
        print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
