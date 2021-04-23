import tensorflow as tf

# boston_housing = tf.keras.datasets.boston_housing
# # train_x和train_y分别接收训练数据集的属性和房价
# # test_x和test_y分别接收测试数据集的属性和房价
# # (train_x,train_y),(test_x,test_y)=boston_housing.load_data()    #默认test_split=0.2，即train数据占比0.8，test数据占比0.2
# (train_x, train_y), (test_x, test_y) = boston_housing.load_data(test_split=0)  # 所有数据划分为训练数据
# 
# print("Training set:", len(train_x))
# print("Testing set:", len(test_x))
# print("Dim of train_x:", train_x.ndim)  # 维度
# print("Dim of train_x:", train_x.shape)  # 形状
# print("Dim of train_y:", train_y.ndim)
# print("Dim of train_y:", train_y.shape)
"""
TensorFlow读取csv文件过程:
1.导入所需的模块并声明全局变量
"""
DATA_FILE = 'boston_housing_data.csv'
BATCH_SIZE = 10
NUM_FEATURES = 14
"""
2.定义一个将文件名作为参数的函数，并返回大小等于 BATCH_SIZE 的张量：
"""


def data_generator(filename):
    """
    3.定义 f_queue 和 reader 为文件名
    :param filename:
    :return:
    """
    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)  # 跳跃第一行属性行
    _, value = reader.read(f_queue)
    """
    4.这里指定要使用的数据以防数据丢失。对 .csv 解码并选择需要的特征。例如，选择 RM、PTRATIO 和 LSTAT 特征
    """
    record_defaults = [[0.0] for _ in range(NUM_FEATURES)]
    # tf.decode_csv将CSV记录转换为张量。每一行映射到一个张量。具有特定类型的“ Tensor”对象的列表。
    # record_defaults: 可接受的类型是`float32`，`float64`，`int32`，`int64`，`string`。输入记录的每列一个张量，其中一个
    # 该列的标量默认值，如果该列是必需的。
    data = tf.decode_csv(value, record_defaults=record_defaults)
    print(data)
    features = tf.stack(tf.gather_nd(data, [[5], [10], [12]]))
    print(features)
    label = data[-1]
    """
    5.定义参数来生成批并使用 tf.train.shuffle_batch() 来随机重新排列张量。该函数返回张量 feature_batch和label_batch：
    """
    # minimum number elements in the queue after a
    dequeuemin_after_dequeue = 10 * BATCH_SIZE  # 100
    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE  # 200
    # shuffle the data to generate BATCH_SIZE sample pairs
    feature_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,
                                                        capacity=capacity,
                                                        min_after_dequeue=dequeuemin_after_dequeue)
    return feature_batch, label_batch


"""
6.这里定义了另一个函数在会话中生成批
"""


def generate_data(feature_batch, label_batch):
    with tf.compat.v1.Session() as sess:
        # 初始化队列线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(5):
            features, labels = sess.run([feature_batch, label_batch])
            # print('Hello', features, labels)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    feature_batch, label_batch = data_generator([DATA_FILE])
    generate_data(feature_batch, label_batch)
