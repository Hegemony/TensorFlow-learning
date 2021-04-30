"""
超参数调整过程
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns

# Data
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
# print(df) # 506 * 13
df['target'] = boston.target  # 加入最后一列，目标列

# 创建测试、训练集
X_train, X_test, y_train, y_test = train_test_split(df[['RM', 'LSTAT', 'PTRATIO']],
                                                    df[['target']], test_size=0.3, random_state=0)

m = len(X_train)
n = 3  # 特征数量
n_hidden = 20  # 隐藏神经元数量
batch_size = 200
eta = 0.01
max_epoch = 1000

"""
1. 调整超参数的第一步是构建模型。与之前一样，在 TensorFlow 中构建模型。
2. 添加一种方法将模型保存在 model_file 中。在 TensorFlow 中，可以使用 Saver 对象来完成。然后保存在会话中：
"""
# saver = tf.train.Saver()
# with tf.compat.v1.Session() as sess:
#     save_path = saver.save(sess, 'tmp/model.ckpt')
#     print('Model saved in file : %s' % save_path)
n_hidden = 30


def get_model():
    return


def mean_squared_error(x, y):
    return


def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 1, activation_fn=tf.sigmoid, scope='out')
    return out


"""
3. 确定要调整的超参数，并为超参数选择可能的值。在这里，你可以做随机的选择、固定间隔值或手动选择。三者分别称为随机搜索、网格搜索和手动搜索。
例如，下面是用来调节学习率的代码。
"""
# 随机选择
learning_rate = np.random.rand(5)
print(learning_rate)

# 网格搜索
learning_rate1 = [i for i in np.arange(0, 1, 0.2)]

# 随机搜索
learning_rate2 = [0.5, 0.6, 0.32, 0.7, 0.01]

"""
4. 选择对损失函数给出最佳响应的参数。所以，可以在开始时将损失函数的最大值定义为 best_loss
（如果是精度，可以选择将自己期望得到的准确率设为模型的最低精度）
"""
best_loss = 2

"""
5. 把你的模型放在 for 循环中，然后保存任何能更好估计损失的模型：
"""
epochs = [50, 60, 70]
batches = [5, 10, 20]
rmse_min = 0.04
for epoch in epochs:
    for batche in batches:
        model = get_model()
        model.compile(loss='mean_square_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch_size, verse=1)
        y_test_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_test_pred)
        if rmse < rmse_min:
            rmse_min = rmse
            model_json = model.to_josn()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    model.save_weights('model.hdf5')
    print('Saved model to disk')