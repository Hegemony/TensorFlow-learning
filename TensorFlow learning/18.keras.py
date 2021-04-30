"""
1.定义模型的类型。Keras 提供了两种类型的模型：序列和模型类 API。Keras 提供各种类型的神经网络层
"""
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()

"""
2. 在 model.add() 的帮助下将层添加到模型中。依照 Keras 文档描述，Keras 提供全连接层的选项（针对密集连接的神经网络）
"""
# Dense(units, activation=None, use_bias=True, )

"""
3. 可以使用它来添加尽可能多的层，每个隐藏层都由前一层提供输入。只需要为第一层指定输入维度
"""
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

"""
4. 一旦模型被定义，需要选择一个损失函数和优化器。Keras 提供了多种损失函数（mean_squared_error、mean_absolute_error、
mean_absolute_percentage_error、categorical_crossentropy 和优化器（sgd、RMSprop、Adagrad、Adadelta、Adam 等）。
损失函数和优化器确定后，可以使用 compile（self，optimizer，loss，metrics=None，sample_weight_mode=None）来配置学习过程：
"""
model.compile(optimizer='rmsprop', loss='categroical_crossentropy', metrics=['accuracy'])
"""
5. 使用 fit 方法训练模型：
"""
# model.fit(data, labels, epochs=10, batch_size=32)

"""
6.可以在 predict 方法 predict(self，x，batch_size=32，verbose=0) 的帮助下进行预测：
"""
# model.predict(test_data, batch_size=10)
