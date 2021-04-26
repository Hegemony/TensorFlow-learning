import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
1. 因为各特征的数据范围不同，需要归一化特征数据。为此定义一个归一化函数。
另外，这里添加一个额外的固定输入值将权重和偏置结合起来。为此定义函数 append_bias_reshape()。该技巧有时可有效简化编程：
"""


def normalize(X):
    """
    Normalize the array X
    :param X:
    :return:
    """
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


def append_bias_reshape(features, labels):
    m = features.shape[0]
    n = features.shape[1]
    """
    np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    """
    x = np.reshape(np.c_[np.ones(m), features], [m, n + 1])
    y = np.reshape(labels, [m, 1])
    return x, y


"""
2.现在使用 TensorFlow contrib 数据集加载波士顿房价数据集，并将其划分为 X_train 和 Y_train。
注意到 X_train 包含所需要的特征。可以选择在这里对数据进行归一化处理，也可以添加偏置并对网络数据重构：
"""
boston = tf.contrib.learn.datasets.load_dataset('boston')
print('boston:', boston.data.shape, boston.target.shape)  # (506, 13), (506, )
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
X_train, Y_train = append_bias_reshape(X_train, Y_train)  # (506, 14), (506, 1)
m = len(X_train)  # 训练样本的数量: 506
n = 13 + 1  # features + bias的数量
print(m)  # 506

"""
3. 为训练数据声明 TensorFlow 占位符。观测占位符 X 的形状变化：
"""
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y')

"""
4.为权重和偏置创建 TensorFlow 变量。通过随机数初始化权重：
"""

w = tf.Variable(tf.random_normal([n, 1]))

"""
5.定义预测的线性回归模型
"""

Y_hat = tf.matmul(X, w)

"""
6. 为了更好地求微分，定义损失函数：
"""
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))

"""
7. 选择优化器
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

"""
8.定义初始化操作符
"""
init_op = tf.global_variables_initializer()
total = []

"""
9. 开始计算图
"""

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    # 训练模型100个epochs
    for i in range(100):
        total_loss = 0
        # for x, y in zip(X_train, Y_train):
        print(X_train.shape, Y_train.shape)
        _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})  # 开始训练
        total_loss += l

        total.append(total_loss / m)
        print('Epoch {0} : Loss {1}'.format(i, total_loss / m))
    w_value = sess.run(w)  # 得到w的值

"""
10. 查看结果
"""
print('w_value', w_value)
# Y_pred = tf.matmul(X_train, w_value)
print('Done')
"""
11. plot the result
"""
# plt.plot(X_train, Y_train, 'bo', label='Real Data')
# plt.plot(X_train, Y_pred, 'r', label='Predicted Data')
# plt.legend()  # 给plt.plot( )中参数label=''传入字符串类型的值，也就是图例的名称
# plt.show()
plt.plot(total)
plt.show()
