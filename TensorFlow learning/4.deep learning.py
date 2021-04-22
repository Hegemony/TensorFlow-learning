import tensorflow as tf

"""
读取数据:
在 TensorFlow 中可以通过三种方式读取数据：
1.通过feed_dict传递数据；
2.从文件中读取数据；
3.使用预加载的数据；
"""
"""
通过feed_dict传递数据
在这种情况下，运行每个步骤时都会使用 run() 或 eval() 函数调用中的 feed_dict 参数来提供数据。这是在占位符的帮助下完成的，
这个方法允许传递 Numpy 数组数据。可以使用 TensorFlow 的以下代码：
"""
y = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32)
with tf.compat.v1.Session() as sess:
    X_Array = tf.random.normal([2, 3], mean=2.0, stddev=2)
    Y_Array = tf.random.normal([2, 3], mean=2.0, stddev=2)
    X_Array = sess.run(X_Array)  # 先sess.run()一下才能传入到下面的feed_dict参数里面
    Y_Array = sess.run(Y_Array)
    Z_Array = tf.add(X_Array, Y_Array)
    print(sess.run(Z_Array, feed_dict={x: X_Array, y: Y_Array}))

"""
从文件中读取：
当数据集非常大时，使用此方法可以确保不是所有数据都立即占用内存（例如 60 GB的 YouTube-8m 数据集）。从文件读取的过程可以通过以下完成：
1、使用字符串张量 ["file0"，"file1"] 或者 [("file%d"i)for i in range(2)] 的方式创建文件命名列表，
   或者使用files=tf.train.match_filenames_once('*.JPG') 函数创建。
2、文件名队列：创建一个队列来保存文件名，此时需要使用 tf.train.string_input_producer 函数：
"""
files = ...
filename_queue = tf.train.string_input_producer(files)
# 这个函数还提供了一个选项来排列和设置批次的最大数量。整个文件名列表被添加到每个批次的队列中。
# 如果选择了 shuffle=True，则在每个批次中都要重新排列文件名。

"""
Reader用于从文件名队列中读取文件。根据输入文件格式选择相应的阅读器。read方法是标识文件和记录（调试时有用）以及标量字符串值的关键字。
例如，文件格式为.csv 时
(csv按行读取，二进制文件按样本的bytes读取，图片一张一张地读取)
"""
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)  # key是文件名，value是默认的内容

"""
Decoder：使用一个或多个解码器和转换操作来将值字符串解码为构成训练样本的张量：
"""
record_defaults = [[1], [1], [1]]
col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)

"""
预加载的数据:
当数据集很小时可以使用，可以在内存中完全加载。因此，可以将数据存储在常量或变量中。在使用变量时，需要将可训练标志设置为 False，
以便训练时数据不会改变。预加载数据为 TensorFlow 常量时：
"""
# 准备预训练的数据作为常量
training_data = ...
training_labels = ...
with tf.compat.v1.Session() as sess:
    x_data = tf.Constant(training_data)
    y_data = tf.Constant(training_labels)

# 准备预训练的数据作为变量
training_data = ...
training_labels = ...
with tf.compat.v1.Session() as sess:
    data_x = tf.placeholder(dtype=training_data.dtype, shape=training_data.shape)
    data_y = tf.placeholder(dtype=training_labels.dtype, shape=training_labels.shape)
    x_data = tf.Variable(data_x, trainable=False, collections=[])  # 设置False，保证训练数据不被改变
    y_data = tf.Variable(data_y, trainable=False, collections=[])

"""
定义模型：
建立描述网络结构的计算图。它涉及指定信息从一组神经元到另一组神经元的超参数、变量和占位符序列以及损失/错误函数。
"""
"""
# 训练/学习：
# 在 DNN 中的学习通常基于梯度下降算法（后续章节将详细讨论），其目的是要找到训练变量（权重/偏置），
# 将损失/错误函数最小化。这是通过初始化变量并使用 run() 来实现的：

评估模型:
一旦网络被训练，通过 predict() 函数使用验证数据和测试数据来评估网络。这可以评价模型是否适合相应数据集，
可以避免过拟合或欠拟合的问题。一旦模型取得让人满意的精度，就可以部署在生产环境中了。
"""

