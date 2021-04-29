"""
导入需要用到的模块：sklearn，该模块可以用来获取数据集，预处理数据，并将其分成训练集和测试集；pandas，
可以用来分析数据集；matplotlib 和 seaborn 可以用来可视化：
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns

"""
2. 加载数据集并创建 Pandas 数据帧来分析数据：
"""
# Data
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
# print(df) # 506 * 13
df['target'] = boston.target  # 加入最后一列，目标列
# print(df) # 506 * 14

"""
3. 了解一些关于数据的细节
"""
print(df.describe())
# count(行数)，mean(平均值)，std(标准差)，min(最小值），
# 25%(第一四分位数)，50%(第二四分位数)，75%(第三四分位数)(返回25%，50%，75%数据量时的数字)，max(最大值)。
#              CRIM          ZN       INDUS  ...           B       LSTAT      target
# count  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000
# mean     3.613524   11.363636   11.136779  ...  356.674032   12.653063   22.532806
# std      8.601545   23.322453    6.860353  ...   91.294864    7.141062    9.197104
# min      0.006320    0.000000    0.460000  ...    0.320000    1.730000    5.000000
# 25%      0.082045    0.000000    5.190000  ...  375.377500    6.950000   17.025000
# 50%      0.256510    0.000000    9.690000  ...  391.440000   11.360000   21.200000
# 75%      3.677083   12.500000   18.100000  ...  396.225000   16.955000   25.000000
# max     88.976200  100.000000   27.740000  ...  396.900000   37.970000   50.000000
#
# [8 rows x 14 columns]

"""
4. 找到输入的不同特征与输出之间的关联
"""
# 画相关性系数图
# plt.subplots()是一个函数，返回一个包含figure和axes对象的元组。
# 因此，使用fig,ax = plt.subplots()将元组分解为fig和ax两个变量。通常，我们只用到ax来控制子图


colormap_, ax = plt.subplots(figsize=(14, 10))
"""
DataFrame.corr(method='pearson', min_periods=1)
data.corr()表示了data中的两个变量之间的相关性，取值范围为[-1,1],取值接近-1，表示反相关，类似反比例函数，取值接近1，表正相关
参数说明：
method：可选值为{‘pearson’, ‘kendall’, ‘spearman’} https://www.jianshu.com/p/7697eb89926a
pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性                                           数据便会有误差。
kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
spearman：非线性的，非正太分析的数据的相关系数
min_periods：样本最少的数据量
返回值：各类型之间的相关系数DataFrame表格。
"""
corr = df.corr(method='pearson')
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9}, ax=ax, annot=True, annot_kws={'fontsize': 12})
plt.show()

# 创建测试、训练集
X_train, X_test, y_train, y_test = train_test_split(df[['RM', 'LSTAT', 'PTRATIO']],
                                                    df[['target']], test_size=0.3, random_state=0)

"""
5. 从前面的代码中，可以看到三个参数 RM、PTRATIO 和 LSTAT 在幅度上与输出之间具有大于 0.5 的相关性。选择它们进行训练。
将数据集分解为训练数据集和测试数据集。使用 MinMaxScaler 来规范数据集。需要注意的一个重要变化是，
由于神经网络使用 Sigmoid 激活函数（Sigmoid 的输出只能在 0～1 之间），所以还必须对目标值 Y 进行归一化：
"""
# 正则化数据
X_train = MinMaxScaler().fit_transform(X_train)
y_train = MinMaxScaler().fit_transform(y_train)
X_test = MinMaxScaler().fit_transform(X_test)
y_test = MinMaxScaler().fit_transform(y_test)

"""
6. 定义常量和超参数：
"""
m = len(X_train)
n = 3  # 特征数量
n_hidden = 20  # 隐藏神经元数量
batch_size = 200
eta = 0.01
max_epoch = 1000

"""
7. 创建一个单隐藏层的多层感知机模型：
"""


def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, n_hidden, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = layers.fully_connected(fc2, 1, activation_fn=tf.sigmoid, scope='out')
    return out


"""
8. 声明训练数据的占位符并定义损失和优化器：
"""
x = tf.placeholder(tf.float32, name='X', shape=[m, n])
y = tf.placeholder(tf.float32, name='Y')
y_hat = multilayer_perceptron(x)
correct_prediction = tf.square(y - y_hat)
loss = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
train = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(max_epoch):
        _, l, p = sess.run([train, loss, y_hat], feed_dict={x: X_train, y: y_train})
        if i % 100 == 0:
            print('Epoch {0}: Loss{1}'.format(i, l))
    # print('Training Done')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Mean Error:', accuracy.eval({x: X_train, y: y_train}))
    plt.scatter(y_train, p)
    plt.show()

