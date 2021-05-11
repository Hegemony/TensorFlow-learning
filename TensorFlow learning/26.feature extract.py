"""
1. 导入处理和显示图像所需的预建模型和附加模块
"""
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
"""
2. 从网络中选择一个特定的图层，并获取输出的特征：
"""
# 预建立和预训练深度学习VGG16模型
base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

# 从block4_pool提取特征
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

"""
3. 提取给定图像的特征
"""
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# 得到特征
features = model.predict(x)
print(features)