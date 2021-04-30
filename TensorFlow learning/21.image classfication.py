"""
1. 导入几个 utils 和核心层用于实现 ConvNet、dropout、fully_connected 和 max_pool。另外，导入一些对图像处理和图像增强有用的模块。
"""
# from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

"""
2. 加载 CIFAR-10 数据，并将其分为 X_train 和 Y_train，X_test 用于测试，Y_test 是测试集的标签。
   对 X 和 Y 进行混洗可能是有用的，因为这样能避免训练依赖于特定的数据配置。最后一步是对 X 和 Y 进行独热编码：
"""
# 数据加载和预处理
from tflearn.datasets import cifar10

(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)  # 随机打乱X, Y
print(X[0].shape)
print(Y[:10])
Y = to_categorical(Y, 10)  # 将训练集转换为二进制编码，也就是onehot编码
Y_test = to_categorical(Y_test, 10)  # 将测试集转换为二进制编码，也就是onehot编码

"""
3. 使用 ImagePreprocessing() 对数据集进行零中心化（即对整个数据集计算平均值），同时进行 STD 标准化（即对整个数据集计算标准差）。
   TFLearn 数据流旨在通过 CPU 先对数据进行预处理，然后在 GPU 上加速模型训练：
"""
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

"""
4. 通过随机左右翻转和随机旋转来增强数据集。这一步是一个简单的技巧，用于增加可用于训练的数据
"""
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

"""
5. 使用之前定义的图片预处理和图片增强操作创建卷积网络。网络由三个卷积层组成。
   第一层有 32 个卷积核，尺寸是 3×3，激活函数用 ReLU，这一层后使用 max_pool 层用于缩小尺寸。
   然后是两个卷积核级联，卷积核的个数是 64，尺寸是 3×3，激活函数是 ReLU
   之后依次是 max_pool 层，具有 512 个神经元、激活函数为 ReLU 的全连接的网络，设置 dropout 概率为 50%。
   最后一层是全连接层，利用 10 个神经元和激活函数 softmax 对 10 个手写数字进行分类。
   请注意，这种特殊类型的 ConvNet 在 CIFAR-10 中非常有效。其中，使用 Adam 优化器（categorical_crossentropy）学习率是 0.001：
"""
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
"""
def conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv2D"):
"""
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

"""
6. 实例化 ConvNet 并以 batch_size=96 训练 50 个 epoch：
"""
# 使用分类器进行训练
model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=96, run_id='cifar10-cnn')