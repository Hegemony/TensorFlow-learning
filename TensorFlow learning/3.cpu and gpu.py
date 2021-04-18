import tensorflow as tf

# sess = tf.InteractiveSession()
"""
TensorFlow 将支持的 CPU 设备命名为“/device：CPU：0”（或“/cpu：0”），第 i 个 GPU 设备命名为“/device：GPU：I”（或“/gpu：I”）。

为了解决这个问题，TensorFlow 可以选择将计算放在一个特定的设备上。默认情况下，如果 CPU 和 GPU 都存在，TensorFlow 会优先考虑 GPU。

TensorFlow 将设备表示为字符串。下面展示如何在 TensorFlow 中指定某一设备用于矩阵乘法的计算。
"""
# 要验证 TensorFlow 是否确实在使用指定的设备（CPU 或 GPU），可以创建会话，并将 log_device_placement 标志设置为 True，即：
# config = tf.ConfigProto(log_device_placement=True)

# 如果你不确定设备，并希望 TensorFlow 选择现有和受支持的设备，则可以将 allow_soft_placement 标志设置为 True：
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# 手动选择 CPU 进行操作：
with tf.device('/cpu:0'):
# with tf.device('/gpu:0'):
    rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a, b)

sess = tf.compat.v1.Session(config=config)  # 指定某个CPU/GPU
sess.run(tf.global_variables_initializer())
print(sess.run(c))
