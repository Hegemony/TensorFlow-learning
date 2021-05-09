"""
1.导入 TensorFlow、tflearn 以及构建网络所需要的模块。然后导入 IMDb 库并执行独热编码和填充：
"""
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

"""
2. 加载数据集，用 0 填充整个句子至句子的最大长度，然后在标签上进行独热编码，其中两个数值分别对应 true 和 false 值。
   请注意，参数 n_words 是词汇表中单词的个数。表外的单词均设为未知。此外，请注意 trainX 和 trainY 是稀疏向量，
   因为每个评论可能仅包含整个单词集的一个子集。
(我们花一点时间来了解数据的格式。数据集经过预处理：每个示例都是一个整数数组，表示电影评论的单词。
每个标签都是0或1的整数值，其中0表示负面评论，1表示正面评论。Training entries: 25000， labels: ：25000

评论文本已转换为整数（在机器学习处理文本中，一般都会根据语料建立一个字库，用词语所在序号来替换掉语料中的文本），
其中每个整数表示字典中的特定单词。
"""
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
print(len(train), len(train[0]), len(train[1]))
print(train[0][:10])
print(train[1][:10])
trainX, trainY = train
testX, testY = test
# 因为是不定长的句子，所以要padding填充,用0填充成一样的长度100，这里填充到后面
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

# 把trainY转成二分类 one-hot编码
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
# print(trainY, testY)

# 如上，预处理过程完成，下面建模其实就很简单。
"""
3. 为数据集中包含的文本构建一个嵌入。就目前而言，考虑这个步骤是一个黑盒子，它把这些词汇映射聚类，以便类似的词汇可能出现在同一个聚类中。
请注意，在之前的步骤中，词汇是离散和稀疏的。通过嵌入操作，这里将创建一个将每个单词嵌入连续密集向量空间的映射。
使用这个向量空间表示将给出一个连续的、分布式的词汇表示。
"""
network = input_data(shape=[None, 100], name='input')
# 这里用embedding层，没有外部的word2vec，输入维是词语的空间，嵌入到128的向量空间
network = tflearn.embedding(network, input_dim=10000, output_dim=128)
"""
4. 创建合适的卷积网络。这里有三个卷积层。由于正在处理文本，这里将使用一维卷积网络，这些图层将并行执行。
每一层需要一个 128 维的张量（即嵌入输出），并应用多个具有有效填充的滤波器（分别为 3、4、5）、激活函数 ReLU 和 L2 regularizer。
然后将每个图层的输出通过合并操作连接起来。接下来添加最大池层，以 50% 的概率丢弃参数的 dropout 层。
最后一层是使用 softmax 激活的全连接层：
"""
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer='L2')
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer='L2')
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer='L2')
# 比较重要的几个参数是inputs, filters, kernel_size，下面分别说明
# inputs :  输入tensor， 维度(None,  a, b) 是一个三维的tensor
#           None  ：  一般是填充样本的个数，batch_size
#           a    ：  句子中的词数或者字数
#           b   :    字或者词的向量维度
# filters :  过滤器的个数
# kernel_size : 卷积核的大小，卷积核其实应该是一个二维的，这里只需要指定一维，是因为卷积核的第二维与输入的词向量维度是一致的，
# 因为对于句子而言，卷积的移动方向只能是沿着词的方向，即只能在列维度移动.

print('branch1:', branch1.shape)
print('0:', network.shape)
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
print('1:', network.shape)  # (100 - 2) + (100 - 3) + (100 - 4)
network = tf.expand_dims(network, 2)
print('2:', network.shape)
# tf.expand_dims：加一个维度
# # 't' is a tensor of shape [2]
# tf.shape(tf.expand_dims(t, 0))  # [1, 2]
# tf.shape(tf.expand_dims(t, 1))  # [2, 1]
# tf.shape(tf.expand_dims(t, -1))  # [2, 1]
#
# # 't2' is a tensor of shape [2, 3, 5]
# tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
# tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
# tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
network = global_max_pool(network)
print('3:', network.shape)
network = dropout(network, 0.5)
print('4:', network.shape)
network = fully_connected(network, 2, activation='softmax')
print('5:', network.shape)

# # 把词向量再输出给LSTM,隐层也是128层，就是词向量的长度
# network = tflearn.lstm(net, 128, dropout=0.8)
"""
5. 学习阶段使用 Adam 优化器以及 categorical_crossentropy 作为损失函数
"""
network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')

"""
6. 在训练中，采用 batch_size=32，观察在训练和验证集上达到的准确度。正如你所看到的，在通过电影评论预测情感表达时能够获得 79% 的准确性
"""
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
