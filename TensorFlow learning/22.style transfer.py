import tensorflow as tf
from tensorflow.python.keras import applications
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras import optimizers

"""
1. 从 Kaggle（https://www.kaggle.com/c/dogs-vs-cats/data）下载狗和猫的数据，并创建一个包含两个子目录（train 和 validation）的数据目录，每个子目录有两个额外的子目录，分别是 dogs 和 cats。
2. 导入稍后将用于计算的 Keras 模块，并保存一些有用的常量：
"""
img_width, img_height = 256, 256
batch_size = 16
epochs = 50
train_data_dir = 'dogs_and_cats/train'
validation_data_dir = 'dogs_and_cats/test'
# 输出
out_categories = 1
# 训练、验证集数量
nb_train_samples = 2000
nb_validation_samples = 100

"""
3. 加载 ImageNet 上预训练的 VGG16 网络，省略最后一层，因为这里将在预建的 VGG16 网络的顶部添加自定义分类网络，并替换原来 VGG16 的分类层：
"""
# 加载在imagenet预训练好的VGG模型
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# print(base_model.layers[:9])
print(base_model.summary())
print(base_model.output_shape)  # (None, 8, 8, 512)
print('*' * 100)

"""
4. 冻结预训练的 VGG16 网络的一定数量的较低层。在这里决定冻结最前面的 15 层
"""
for layer in base_model.layers[:15]:
    layer.trainable = False

"""
5. 为了分类，添加一组自定义的顶层
"""
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(out_categories, activation='sigmoid'))

"""
6. 自定义网络应该单独进行预训练，为了简单起见，这里省略了这部分，将此任务交给读者：
"""

"""
7. 创建一个新的网络，这是预训练的 VGG16 网络和预训练的定制网络的组合体：
"""
# 得到最终模型，组合预训练的和自己定义的
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])

"""
8. 重新训练组合的新模型，仍然保持 VGG16 的 15 个最低层处于冻结状态。在这个特定的例子中，也使用 Image Augumentator 来增强训练集：
"""
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=nb_validation_samples // batch_size,
    verbose=2, workers=12
)
"""
9. 在组合网络上评估结果
"""
score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
scores = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
# print(score, scores)