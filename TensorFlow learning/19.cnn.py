import tensorflow as tf

# 输入是1张 3*3 大小的图片，图像通道数是5，
# 卷积核是 3*3 大小，通道数量是5，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,1,1,1] 的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))

conv1 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=None, data_format=None,
                     name=None)
"""
卷积层参数说明如下：
--input：张量，必须是 half、float32、float64 三种类型之一。
--filter：张量必须具有与输入相同的类型。
--strides：整数列表。长度是 4 的一维向量。输入的每一维度的滑动窗口步幅。必须与指定格式维度的顺序相同。
--padding：可选字符串为 SAME、VALID。要使用的填充算法的类型。
--use_cudnn_on_gpu：一个可选的布尔值，默认为 True。
--data_format：可选字符串为 NHWC、NCHW，默认为 NHWC。指定输入和输出数据的数据格式。使用默认格式 NHWC，数据按照以下顺序存储：[batch，in_height，in_width，in_channels]。或者，格式可以是 NCHW，数据存储顺序为：[batch，in_channels，in_height，in_width]。
--name：操作的名称（可选）。
"""

# 一个简单而通用的选择是所谓的最大池化算子，它只是输出在区域中观察到的最大输入值。
# 在 TensorFlow 中，如果想要定义一个大小为 2×2 的最大池化层，可以这样写：

# h : 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch_size, height, width, channels]这样的shape
# k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
# strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
# padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加

value = tf.Variable(tf.random_normal([1, 3, 3, 5]))
ksize = tf.Variable(tf.random_normal([1, 2, 2, 1]))
pool1 = tf.nn.max_pool(value, ksize, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW', name=None)
"""
池化层参数说明如下：
value：形状为 [batch，height，width，channels] 和类型是 tf.float32 的四维张量。
ksize：长度 >=4 的整数列表。输入张量的每个维度的窗口大小。
strides：长度 >=4 的整数列表。输入张量的每个维度的滑动窗口的步幅。
padding：一个字符串，可以是 VALID 或 SAME。
data_format：一个字符串，支持 NHWC 和 NCHW。
name：操作的可选名称。
"""
