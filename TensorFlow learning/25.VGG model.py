# """
# 1.导入 matplotlib 和 keras-vis 使用的模块。另外还需要载入预建的 VGG16 模块。Keras 可以轻松处理这个预建网络
# """
# import tensorflow as tf
# import matplotlib.pyplot as plt
# # from vis.utils import utils
# # from vis.visualization import visualize_activation
#
# """
# 2. 通过使用 Keras 中包含的预构建图层获取 VGG16 网络，并使用 ImageNet 权重进行训练：
# """
# model = tf.python.keras.applications.VGG16(weights='imagenet', include_top=True)
# model.summary()
# print('Model loaded.')
"""
导入tensorflow 以及numpy库
"""
import numpy as np
import tensorflow as tf

"""
定义各个层的结构
"""


def get_weight_variable(shape):
    '''
    生成weight，初始化'''
    return tf.get_variable('weight', shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias_variable(shape):
    '''bias'''
    return tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(0))


def conv2d(x, w, padding='same', s=1):
    '''卷积层'''
    x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=padding)
    return x


def maxPoolLayer(x):
    '''池化层'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d_layer(x, in_chs, out_chs, ksize, layer_name):
    '''卷积层'''
    with tf.variable_scope(layer_name):
        w = get_weight_variable([ksize, ksize, in_chs, out_chs])
        b = get_bias_variable([out_chs])
        y = tf.nn.relu(tf.nn.bias_add(conv2d(x, w, padding='SAME', s=1), b))
    return y


def fc_layer(x, in_kernels, out_kernels, layer_name):
    '''全连接层'''
    with tf.variable_scope(layer_name):
        w = get_weight_variable([in_kernels, out_kernels])
        b = get_bias_variable([out_kernels])
        y = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b))
    return y


"""
VGG16架构
"""


def VGG16(x):
    # conv1_1 = conv2d_layer(x, tf.get_shape(x).as_list()[-1], 64, 3, 'conv1_1')
    conv1_1 = conv2d_layer(x, 3, 64, 3, 'conv1_1')
    conv1_2 = conv2d_layer(conv1_1, 64, 64, 3, 'conv1_2')
    pool_1 = maxPoolLayer(conv1_2)
    ''' 输入为224*224*3 输出224*224*64，池化-112*112*64'''
    conv2_1 = conv2d_layer(pool_1, 64, 128, 3, 'conv2_1')
    conv2_2 = conv2d_layer(conv2_1, 128, 128, 3, 'conv2_2')
    pool2 = maxPoolLayer(conv2_2)
    '''通道数变为128，池化后由112变为112/2=56'''
    conv3_1 = conv2d_layer(pool2, 128, 256, 3, 'conv3_1')
    conv3_2 = conv2d_layer(conv3_1, 256, 256, 3, 'conv3_2')
    conv3_3 = conv2d_layer(conv3_2, 256, 256, 3, 'conv3_3')
    pool3 = maxPoolLayer(conv3_3)
    '''通道变为256，池化后/2变为28'''
    conv4_1 = conv2d_layer(pool3, 256, 512, 3, 'conv4_1')
    conv4_2 = conv2d_layer(conv4_1, 512, 512, 3, 'conv4_2')
    conv4_3 = conv2d_layer(conv4_2, 512, 512, 3, 'conv4_3')
    pool4 = maxPoolLayer(conv4_3)
    '''通道变为512，池化后变为14'''
    conv5_1 = conv2d_layer(pool4, 512, 512, 3, 'conv5_1')
    conv5_2 = conv2d_layer(conv5_1, 512, 512, 3, 'conv5_2')
    conv5_3 = conv2d_layer(conv5_2, 512, 512, 3, 'conv5_3')
    pool5 = maxPoolLayer(conv5_3)
    '''池化后变为7'''
    pool5_flatten_dims = int(np.prod(pool5.get_shape().as_list()[1:]))
    pool5_flatten = tf.reshape(pool5, [-1, pool5_flatten_dims])
    '''最后将7*7*512展平'''
    fc_6 = fc_layer(pool5_flatten, pool5_flatten_dims, 4096, 'fc6')
    '''将7*7*512接一个全连接层（7*7*512*4096）变为1*4096'''
    fc_7 = fc_layer(fc_6, 4096, 4096, 'fc7')

    fc_8 = fc_layer(fc_7, 4096, 1000, 'fc8')
    '''最后输出为1*1000的结果'''
    return fc_8


n_classes = 10
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
y = tf.placeholder(tf.int64, shape=[None, n_classes], name='label')

output = VGG16(x)
print(output)