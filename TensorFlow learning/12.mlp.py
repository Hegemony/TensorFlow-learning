import tensorflow as tf
import numpy as np

"""
1. 定义要使用的超参数
"""
eta = 0.4
epsilon = 1e-4
max_epochs = 100


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


"""
2. 指定训练数据。在这个例子中，取三个输入神经元（A，B，C）并训练它学习逻辑 AB+BC
"""
T, F = 1.0, 0.0
X_in = [
    [T, T, T, T],
    [T, T, F, T],
    [T, F, T, T],
    [T, F, F, T],
    [F, T, T, T],
    [F, T, F, T],
    [F, F, T, T],
    [F, F, F, T]
]
Y = [
    [T],
    [T],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F]
]

"""
3. 定义要用到的变量和用于计算更新的计算图，最后执行计算图
"""
W = tf.Variable(tf.random_normal([4, 1], stddev=2, seed=0))
h = tf.matmul(X_in, W)
Y_hat = threshold(h)
error = Y - Y_hat
mean_error = tf.reduce_mean(tf.square(error))

"""
matmul(a, b, transpose_a=False, transpose_b=False, 
    adjoint_a=False, adjoint_b=False, a_is_sparse=False,
    b_is_sparse=False, name=None)
--transpose_a：如果 True,a 在乘法之前转置.
--transpose_b：如果 True,b 在乘法之前转置.
"""
dw = eta * tf.matmul(X_in, error, transpose_a=True)  # 求梯度
"""
tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)，
--函数功能是将value赋值给ref
--ref必须是tf.Variable创建的tensor，如果ref=tf.constant()就会报错，而且默认情况下ref的shape和value的shape是相同的
"""
train = tf.assign(W, W + dw)
init = tf.global_variables_initializer()
err = 1
epoch = 0

with tf.compat.v1.Session() as sess:
    sess.run(init)
    while err > epsilon and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mean_error, train])
        print('epoch:{0} mean error:{1}'.format(epoch, err))
    print('complete')