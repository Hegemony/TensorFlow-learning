import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
1.在神经网络中，所有的输入都线性增加。为了使训练有效，输入应该被归一化，所以这里定义一个函数来归一化输入数据：
"""


def normalize(x):
    """
    进行正则化
    :param x:
    :return:
    """
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std
    return x


"""
2. 现在使用 TensorFlow contrib 数据集加载波士顿房价数据集，并将其分解为 X_train 和 Y_train。可以对数据进行归一化处理：
"""

boston = tf.contrib.learn.datasets.load_dataset('boston')
print('boston:', boston.data.shape, boston.target.shape)  # (506, 13), (506, )
X_train, Y_train = boston.data[:, 5], boston.target
# X_train = normalize(X_train)
print('X_train', X_train.shape)  # (506,)
print('Y_train', Y_train.shape)  # (506,)
n_samples = len(X_train)
print(n_samples)  # 506
"""
3. 为训练数据声明 TensorFlow 占位符
"""
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

"""
4. 创建TensorFlow的权重和偏置变量且初始值为零
"""
b = tf.Variable(0.0)
w = tf.Variable(0.0)

"""
5. 定义预测的线性回归模型
"""
Y_hat = X * w + b

"""
6. 定义损失函数
"""
loss = tf.square(Y - Y_hat, name='loss')

"""
7. 选择梯度下降优化器
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

"""
8. 声明初始化操作符
"""
init_op = tf.global_variables_initializer()
total = []

"""
9. 现在，开始计算图，训练100次：
"""

with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(init_op)
    # writer = tf.summary.FileWriter('graphs', sess.graph)
    # 训练模型100个epochs
    for i in range(100):
        total_loss = 0
        for x, y in zip(X_train, Y_train):
            print(x.shape, y.shape, X_train.shape, Y_train.shape)
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})  # 开始训练
            total_loss += l

        total.append(total_loss / n_samples)
        print('Epoch {0} : Loss {1}'.format(i, total_loss / n_samples))
        b_value, w_value = sess.run([b, w])  # 得到b, w的值

"""
10. 查看结果
"""
print('b_value', b_value)
print('w_value', w_value)
Y_pred = X_train * w_value + b_value
print('Done')
"""
11. plot the result
"""
plt.plot(X_train, Y_train, 'bo', label='Real Data')
plt.plot(X_train, Y_pred, 'r', label='Predicted Data')
plt.legend()  # 给plt.plot( )中参数label=''传入字符串类型的值，也就是图例的名称
plt.show()
plt.plot(total)
plt.show()
