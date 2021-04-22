import tensorflow as tf

"""
按照损失函数的负梯度成比例地对系数（W 和 b）进行更新。根据训练样本的大小，有三种梯度下降的变体：
1.Vanilla 梯度下降：在 Vanilla 梯度下降（也称作批梯度下降）中，在每个循环中计算整个训练集的损失函数的梯度。
  该方法可能很慢并且难以处理非常大的数据集。该方法能保证收敛到凸损失函数的全局最小值，但对于非凸损失函数可能会稳定在局部极小值处。
2.随机梯度下降：在随机梯度下降中，一次提供一个训练样本用于更新权重和偏置，从而使损失函数的梯度减小，然后再转向下一个训练样本。
  整个过程重复了若干个循环。由于每次更新一次，所以它比 Vanilla 快，但由于频繁更新，所以损失函数值的方差会比较大。
3.小批量梯度下降：该方法结合了前两者的优点，利用一批训练样本来更新参数。
"""
# 定义计算图
m = 4
n = 2
p = 3
X = tf.placeholder(tf.float32, name='X', shape=[m, n])  # 占位符先占位
Y = tf.placeholder(tf.float32, name='Y', shape=[m, p])
X_data = tf.random.normal([4, 2], mean=2.0, stddev=2)  # 传入的数据的定义
Y_data = tf.random.normal([4, 3], mean=2.0, stddev=2)

w0 = tf.Variable(tf.zeros([1, p]), name='bias')
w1 = tf.Variable(tf.random_normal([n, 1]), name='weights')

# 线性模型
Y_hat = tf.matmul(X, w1) + w0

loss = tf.square(Y - Y_hat, name='Loss')

"""
优化器选择：
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 可以使用 tf.train.AdadeltaOptimizer() 来实现一个自适应的、单调递减的学习率，它使用两个初始化参数 learning_rate和衰减因子rho：
optimizer1 = tf.train.AdadeltaOptimizer(learning_rate=0.8, rho=0.95).minimize(loss)
# TensorFlow 也支持 Hinton 的 RMSprop，其工作方式类似于 Adadelta 的 tf.train.RMSpropOptimizer()：
optimizer2 = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.8, momentum=0.1).minimize(loss)
# 另一种 TensorFlow 支持的常用优化器是 Adam 优化器。该方法利用梯度的一阶和二阶矩对不同的系数计算不同的自适应学习率：
optimizer3 = tf.train.AdamOptimizer().minimize(loss)
train_step = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # placeholder在sess.run()的时候传入值
    X_data = sess.run(X_data)  # 传入的数据的赋值
    Y_data = sess.run(Y_data)
    # sess.run(train_step, feed_dict={X: X_data, Y: Y_data})
    for i in range(10):
        # #使用Feed操作，此步执行训练操作的op，将数据喂给他, 这里可以打印loss的值，但是loss不是必要的，需要train_op才可以执行
        print(sess.run([loss, train_step], feed_dict={X: X_data, Y: Y_data}))


"""
根据 TensorFlow 文档，在训练模型时，通常建议在训练过程中降低学习率。该函数利用指数衰减函数初始化学习率。
需要一个 global_step 值来计算衰减的学习率。可以传递一个在每个训练步骤中递增的 TensorFlow 变量。函数返回衰减的学习率。

变量：
learning_rate：标量float32或float64张量或者Python数字。初始学习率。
global_step：标量int32或int64张量或者Python数字。用于衰减计算的全局步数，非负。
decay_steps：标量int32或int64张量或者Python数字。正数，参考之前所述的衰减计算。
decay_rate：标量float32或float64张量或者Python数字。衰减率。
staircase：布尔值。若为真则以离散的间隔衰减学习率。
name：字符串。可选的操作名。默认为ExponentialDecay。

返回：
与learning_rate类型相同的标量张量。衰减的学习率。
实现指数衰减学习率的代码如下：
tf.train.exponential_decay(
    learning_rate,初始学习率
    global_step,当前迭代次数
    decay_steps,衰减速度（在迭代到该次数时学习率衰减为earning_rate * decay_rate）
    decay_rate,学习率衰减系数，通常介于0-1之间。
    staircase=False,(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
    name=None
)
"""
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.2
# lr指数衰减
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=10000, decay_rate=0.95, staticase=True)
