import tensorflow as tf

"""
# tf.InteractiveSession()是一种交互式的session方式，它让自己成为了默认的session，
# 也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处
# 这样的话就是run()和eval()函数可以不指明session啦。
"""
sess = tf.InteractiveSession()

# 创建一个单位矩阵
I_matrix = tf.eye(5)
print(I_matrix.eval())

x = tf.Variable(tf.eye(10))
sess.run(tf.global_variables_initializer())
# x.initializer.run()

# 若有一个 x是Tensor对象，调用x.eval()相当于调用sess.run(x)
# 两者的区别主要在于，eval一次只能得到一个结点的值，而run可以得到多个。
# float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
# t = float_tensor * float_tensor
#
# sess = tf.Session()
# with sess.as_default():
#     print(t.eval(), float_tensor.eval())
#     print(sess.run((t, float_tensor)))
print(x.eval())

# 创建一个5 * 10的随机矩阵
A = tf.Variable(tf.random_normal([5, 10]))
sess.run(tf.global_variables_initializer())
# A.initializer.run()
print(A.eval())

# 两个矩阵相乘
product = tf.matmul(A, x)
print('product:', product.eval())

# tensorflow类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
b = tf.Variable(tf.random_uniform([5, 10], 0, 2, dtype=tf.int32))
sess.run(tf.global_variables_initializer())
b_new = tf.cast(b, dtype=tf.float32)
print('b_new', b_new.eval())

"""
一些其他有用的矩阵操作，如按元素相乘、乘以一个标量、按元素相除、按元素余数相除等，可以执行如下语句：
"""
# 创建两个随机矩阵
a = tf.Variable(tf.random_normal([4, 5], stddev=2))
b = tf.Variable(tf.random_normal([4, 5], stddev=2))

# 逐元素相乘
A = a * b
# 用标量2相乘A
B = tf.scalar_mul(2, A)
# 逐元素相除
C = tf.div(a, b)
# 逐元素模
D = tf.mod(a, b)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a, b, A_R, B_R, C_R, D_R = sess.run([a, b, A, B, C, D])
    print(a, b, A_R, B_R, C_R, D_R)