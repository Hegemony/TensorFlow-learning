import tensorflow as tf

# m 是样本数量，n 是特征数量，P 是类别数量。
m = 1000
n = 15
p = 2
# 在标准线性回归的情况下，只有一个输入变量和一个输出变量：
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

# 线性模型
Y_hat = X * w1 + w0

loss = tf.square(Y - Y_hat, name='Loss')

# 多元线性模型
# 在多元线性回归的情况下，输入变量不止一个，而输出变量仍为一个。现在可以定义占位符X的大小为 [m，n]，其中 m 是样本数量，n 是特征数量，
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y')

w0 = tf.Variable(0.0)
w1 = tf.Variable([n, 1])

Y_hat = tf.matmul(X, w1) + w0

loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))
# 在逻辑回归的情况下，损失函数定义为交叉熵。输出 Y 的维数等于训练数据集中类别的数量，其中 P 为类别数量：
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y', shape=[m, p])

w0 = tf.Variable(tf.zeros([1, p]), name='bias')
w1 = tf.Variable(tf.random_normal([n, 1]), name='weights')

Y_hat = tf.matmul(X, w1) + w0

entropy = tf.nn.softmax_cross_entropy_with_logits(Y_hat, Y)
loss = tf.reduce_mean(entropy)

# 加入L1正则项
lamda = tf.constant(0.8)
regularization_param = lamda * tf.reduce_sum(tf.abs(w1))

# New loss
loss += regularization_param

# 加入L2正则项
lamda = tf.constant(0.8)
regularization_param = lamda * tf.nn.l2_loss(w1)

# New loss
loss += regularization_param

"""
tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和;(默认计算所有的数值)
tf.reduce_max  :  计算tensor指定轴方向上的各个元素的最大值;
tf.reduce_all :  计算tensor指定轴方向上的各个元素的逻辑和（and运算）;
tf.reduce_any:  计算tensor指定轴方向上的各个元素的逻辑或（or运算）;
"""
