import tensorflow as tf

# 由于要打印的信息是一个常量字符串，因此使用 tf.constant：
message = tf.constant('Hello World')

# 为了执行计算图，利用 with 语句定义 Session，并使用 run 来运行：
# with tf.Session() as sess:  # old版本
with tf.compat.v1.Session() as sess:  # new版本
    print(sess.run(message).decode())

"""
TensorFlow 程序解读分析
前面的代码分为以下三个主要部分：
第一部分 import 模块包含代码将使用的所有库，在目前的代码中只使用 TensorFlow，其中语句 import tensorflow as tf 则允许 Python 访问 TensorFlow 所有的类、方法和符号。
第二个模块包含图形定义部分...创建想要的计算图。在本例中计算图只有一个节点，tensor 常量消息由字符串“Hello World”构成。
第三个模块是通过会话执行计算图，这部分使用 with 关键字创建了会话，最后在会话中执行以上计算图。
"""

"""
tensorflow 程序结构
"""
v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v1, v2)

with tf.compat.v1.Session() as sess:
    print(sess.run(v_add))

"""
tensorflow 常量、变量和占位符
"""

# 声明一个常量(标量)
t1 = tf.constant(4)
# 声明一个向量
t2 = tf.constant([4, 3, 2])

# 定义所有元素为0的张量, 创建一个[M, N]的零元素矩阵，数据类型可以为int32, float64
# tf.zeros([M, N], tf.dtype)
t_zero = tf.zeros([2, 3], tf.int32)

# 定义所有元素为1的张量, 创建一个[M, N]的元素均为1的矩阵，数据类型可以为int32, float64
t_one = tf.ones([3, 3], tf.float64)

with tf.compat.v1.Session() as sess:
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(t_zero))
    print(sess.run(t_one))

"""
更进一步，还有以下语句：
在一定范围内生成一个从初值到终值等差排布的序列：
"""
# tf.linspace(start, stop, num)
# 相应的值为 (stop-start)/(num-1)。

linspace_t = tf.linspace(2.0, 5.0, 5)

# 从开始（默认值=0）生成一个数字序列，增量为 delta（默认值=1），直到终值（但不包括终值）：
# tf.range(start,limit,delta)
range_t = tf.range(10)

with tf.compat.v1.Session() as sess:
    print(sess.run(linspace_t))
    print(sess.run(range_t))

"""
TensorFlow 允许创建具有不同分布的随机张量：
1. 使用以下语句创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组：
tf.random_normal是没有要求一定在（μ-2σ，μ+2σ）之内的
"""
t_random = tf.random_normal([2, 3], mean=2.0, stddev=4, seed=12)

"""
2. 创建一个具有一定均值和标准差，形状为[M, N]的截尾正态分布随机数组：
虽然是输出正态分布，但是它生成的值是在距离均值两个标准差范围之内的（就是tf.truncated_normal要求一定在（μ-2σ，μ+2σ）之内的）
"""
t_random1 = tf.truncated_normal([1, 5], stddev=2, seed=12)

"""
3. 要在种子的 [minval（default=0），maxval（default=1）] 范围内创建形状为 [M，N] 的给定伽马分布随机数组，请执行如下语句：
"""
t_random2 = tf.random_uniform([2, 3], maxval=4, seed=12)

"""
4.要将给定的张量随机裁剪为指定的大小，使用以下语句：
tf.random_crop(t_random,[2,5],seed=12)

这里，t_random 是一个已经定义好的张量。这将导致随机从张量 t_random 中裁剪出一个大小为 [2，5] 的张量。

很多时候需要以随机的顺序来呈现训练样本，可以使用 tf.random_shuffle() 来沿着它的第一维随机排列张量。
如果 t_random 是想要重新排序的张量，使用下面的代码：
tf.random_shuffle(t_random)
"""
t_random3 = tf.random_crop(t_random2, [2, 2], seed=12)

"""
5.随机生成的张量受初始种子值的影响。要在多次运行或会话中获得相同的随机数，应该将种子设置为一个常数值。当使用大量的随机张量时，
可以使用 tf.set_random_seed() 来为所有随机产生的张量设置种子。以下命令将所有会话的随机张量的种子设置为 54：
"""
tf.set_random_seed(54)

with tf.compat.v1.Session() as sess:
    print('random:', sess.run(t_random))
    print('random1:', sess.run(t_random1))
    print('random2:', sess.run(t_random2))
    print('random3:', sess.run(t_random3))

"""
TensorFlow变量：
它们通过使用变量类来创建。变量的定义还包括应该初始化的常量/随机值。下面的代码中创建了两个不同的张量变量 t_a 和 t_b。
两者将被初始化为形状为 [50，50] 的随机均匀分布，最小值=0，最大值=10：
"""
rand_t0 = tf.random_uniform([50, 50], 0, 10, seed=0)
t_a = tf.Variable(rand_t0)
t_b = tf.Variable(rand_t0)
print(t_a, t_b)
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    """
    一定要加上这句，官方解释是初始化模型的参数。
    在TensorFlow的世界里，变量的定义和初始化是分开的，一开始，tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程。
    因为tf.Variable 生成的是一个张量，那么 name 就是一个张量的名字，如果你不主动声明的话，就是默认的 Variable
    而如果你要得到，变量的值的话，那么你就需要对张量进行计算，首先对变量进行初始化，使用会话进行计算:
    sess.run(tf.global_variables_initializer())
    """
    print('t_a:', sess.run(t_a))
    print('t_b:', sess.run(t_b))

weights = tf.Variable(tf.random_normal([100, 100], stddev=2))
bias = tf.Variable(tf.zeros([100]), name='biases')

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    """
    一定要加上这句，官方解释是初始化模型的参数。
    在TensorFlow的世界里，变量的定义和初始化是分开的，一开始，tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程(也就是定义)。
    因为tf.Variable 生成的是一个张量，那么 name 就是一个张量的名字，如果你不主动声明的话，就是默认的 Variable
    而如果你要得到，变量的值的话，那么你就需要对张量进行计算，首先对变量进行初始化，使用会话进行计算（这步骤是初始化）:
    sess.run(tf.global_variables_initializer())
    """
    print('weights:', sess.run(weights))
    print('bias:', sess.run(bias))

"""
TensorFlow 占位符
介绍完常量和变量之后，我们来讲解最重要的元素——占位符，它们用于将数据提供给计算图。可以使用以下方法定义一个占位符：
tf.placeholder(dtype,shape=None,name=None)
dtype 定占位符的数据类型，并且必须在声明占位符时指定。在这里，为 x 定义一个占位符并计算 y=2*x，使用 feed_dict输入一个随机的 4×5 矩阵：
"""
x = tf.placeholder('float')
y = 2 * x
data = tf.random_uniform([4, 5], 10)

with tf.compat.v1.Session() as sess:
    x_data = sess.run(data)
    print(x_data)
    print(sess.run(y, feed_dict={x: x_data}))

"""
需要注意的是，所有常量、变量和占位符将在代码的计算图部分中定义。如果在定义部分使用 print 语句，只会得到有关张量类型的信息，而不是它的值。

为了得到相关的值，需要创建会话图并对需要提取的张量显式使用运行命令，如下所示：
print(sess.run(t_1))
"""

"""
sess.run(fetches，feed_dict):
这个是让fetches节点动起来，告诉tensorflow，想要此节点的输出。

fetches 可以是list或者tensor向量

feed_dict。替换原图中的某个tensor

"""