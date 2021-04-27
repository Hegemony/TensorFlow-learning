import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def threshold(x):
    """
    tf.less(x, y, name=None) 如果同一位置(x < y)的则输出True, 否则输出false.
    注意：Less支持广播.
    :param x:
    :return:
    """
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    """
    tf.where(condition, x=None, y=None, name=None)
    作用：该函数的作用是根据condition, 返回相对应的x或y, 返回值是一个tf.bool类型的Tensor。
    若condition=True,则返回对应X的值，若为False则返回对应的Y值。
    """
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


# plot threshold activation function
h = np.linspace(-1, 1, 50)  # (-1, 1)区间划分50步
# print(h)
out = threshold(h)

init = tf.global_variables_initializer()

"""
1.阈值激活函数：这是最简单的激活函数。在这里，如果神经元的激活值大于零，那么神经元就会被激活；
否则，它还是处于抑制状态。下面绘制阈值激活函数的图，随着神经元的激活值的改变在 TensorFlow 中实现阈值激活函数：

"""
with tf.compat.v1.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    # print(y)
    """
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    """
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Threshold Activation Function')
    plt.plot(h, y)
    plt.show()

"""
2. Sigmoid
"""
# plot sigmoid activation function
h = np.linspace(-10, 10, 50)  # (-1, 1)区间划分50步
# print(h)
out = tf.sigmoid(h)
init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Sigmoid Activation Function')
    plt.plot(h, y)
    plt.show()

"""
3. Tanh
"""
# plot tanh activation function
h = np.linspace(-10, 10, 50)  # (-1, 1)区间划分50步
# print(h)
out = tf.tanh(h)

init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Tanh Activation Function')
    plt.plot(h, y)
    plt.show()

"""
4. Relu
"""
# plot Relu activation function
h = np.linspace(-10, 10, 50)  # (-1, 1)区间划分50步
# print(h)
out = tf.nn.relu(h)

init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Relu Activation Function')
    plt.plot(h, y)
    plt.show()

"""
5. Softmax
"""
# plot softmax activation function
h = np.linspace(-10, 10, 50)  # (-1, 1)区间划分50步
# print(h)
out = tf.nn.softmax(h)

init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    # print(y)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Softmax Activation Function')
    plt.plot(h, y)
    plt.show()


"""
1. 阈值激活函数用于 McCulloch Pitts 神经元和原始的感知机。这是不可微的，在 x=0 时是不连续的。
   因此，使用这个激活函数来进行基于梯度下降或其变体的训练是不可能的。
2. Sigmoid 激活函数一度很受欢迎，从曲线来看，它像一个连续版的阈值激活函数。它受到梯度消失问题的困扰，即函数的梯度在两个边缘附近变为零。
   这使得训练和优化变得困难。
3. 双曲正切激活函数在形状上也是 S 形并具有非线性特性。该函数以 0 为中心，与 Sigmoid 函数相比具有更陡峭的导数。
   与 Sigmoid 函数一样，它也受到梯度消失问题的影响。线性激活函数是线性的。该函数是双边都趋于无穷的 [-inf，inf]。
   它的线性是主要问题。线性函数之和是线性函数，线性函数的线性函数也是线性函数。因此，使用这个函数，不能表示复杂数据集中存在的非线性。
4. ReLU 激活函数是线性激活功能的整流版本，这种整流功能允许其用于多层时捕获非线性。
   使用 ReLU 的主要优点之一是导致稀疏激活。在任何时刻，所有神经元的负的输入值都不会激活神经元。就计算量来说，这使得网络在计算方面更轻便。
5. ReLU 神经元存在死亡 ReLU 的问题，也就是说，那些没有激活的神经元的梯度为零，因此将无法进行任何训练，并停留在死亡状态。
   尽管存在这个问题，但 ReLU 仍是隐藏层最常用的激活函数之一。Softmax 激活函数被广泛用作输出层的激活函数，该函数的范围是 [0，1]。
   在多类分类问题中，它被用来表示一个类的概率。所有单位输出和总是 1。
"""
