"""
1. 导入 numpy 进行数值计算，functools 定义一个或多个参数已经填充的偏函数，Pillow 用于图像处理，matplotlib 用于产生图像：
"""
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

"""
2. 设置内容图像和预训练模型的路径。从随机噪声的种子图像开始：
"""
content = '.jpg'
img_noise = np.random.uniform(size=(224, 224, 3) + 100.0)
model_fn = 'tensorflow_inception_graph.pb'

"""
3. 以 graph 的形式加载从网上下载的 Inception 网络。初始化一个 TensorFlow 会话，用 FastGFile(..) 加载这个 graph，
   并用 ParseFromstring(..) 解析该 graph。之后，使用 placeholder(..) 方法创建一个占位符作为输入。 
   imagenet_mean 是预先计算的常数，这里的内容图像减去该值以实现数据标准化。
   事实上，这是训练得到的平均值，规范化使得收敛更快。该值将从输入中减去并存储在 t_preprocessed 变量中，然后用于加载 graph 定义：
"""
# 加载图
graph = tf.Graph()
sess = tf.InteractiveSession()
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')  # define input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

"""
4. 定义一些 util 函数来可视化图像
"""


def showarray(a):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    plt.imshow()
    plt.show()


def visstd(a, s=0.1):
    """
    Normalize the image range for  visualization
    :param a:
    :param s:
    :return:
    """
    return (a - a.mean()) / max(a.std, 1e-4) * s + 0.5


def T(layer):
    """
    得到输出层的tensor
    :param layer:
    :return:
    """
    return graph.get_tensor_by_name('import/%s:0' % layer)


def tffunc(*argtypes):
    """
    转换TF-graph 生成function变成一个有规律的
    :param argtypes:
    :return:
    """
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


resize = tffunc(np.float32, np.int32)(resize)

"""
5. 计算图像的梯度上升值。为了提高效率，应用平铺计算，其中在不同的图块上计算单独的梯度上升。
   通过多次迭代对图像应用随机偏移以模糊图块的边界
"""


def calc_grad_tiled(img, t_grad, title_size=512):
    """
    计算一张图像上梯度的值，
    :param img:
    :param t_grad:
    :param title_size:
    :return:
    """
