import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义数据集路径和图像参数
DATA_DIR = 'data/' # 你的 train/test 文件夹所在的上级目录
IMAGE_SIZE = (48, 48) # FER-2013 图像的标准尺寸
BATCH_SIZE = 64 # 加载数据时的批次大小

# 2. 使用 ImageDataGenerator 加载数据
# 由于数据已经是灰度图，且我们希望在预处理阶段进行归一化 (0-255 -> 0-1)
datagen = ImageDataGenerator(rescale=1./255)

# 加载训练集
train_generator = datagen.flow_from_directory(
    directory=DATA_DIR + 'train',
    target_size=IMAGE_SIZE,
    color_mode='grayscale', # 图像是灰度的
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 类别标签使用独热编码
    shuffle=True
)

# 加载测试集
test_generator = datagen.flow_from_directory(
    directory=DATA_DIR + 'test',
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # 测试集通常不打乱
)

print("-" * 30)
print("情绪标签对应的索引 (Keras 自动生成):")
print(train_generator.class_indices)
print("-" * 30)

# 3. 数据探索：检查样本分布 (由 flow_from_directory 打印的结果就是分布)
# 在运行 train_generator.flow_from_directory 时，它已经打印了每个类别的样本数，完成了探索任务！
# 例如：Found 28709 images belonging to 7 classes.

# 4. 数据探索：预览一张图片
# 从生成器中取出一批数据 (X_batch: 图像, y_batch: 标签)
X_batch, y_batch = next(train_generator)
image = X_batch[0] # 取出第一张图片
label_index = np.argmax(y_batch[0]) # 获取独热编码的索引

# 反转字典，根据索引获取标签名
idx_to_label = {v: k for k, v in train_generator.class_indices.items()}
label_name = idx_to_label[label_index]

plt.figure(figsize=(4, 4))
# 由于图像已被归一化到 0-1，imshow 可以直接显示
plt.imshow(image.squeeze(), cmap='gray') 
plt.title(f'Sample Image - Emotion: {label_name}')
plt.axis('off')
plt.show()

# 5. 最终数据形状确认
# X_batch.shape: (64, 48, 48, 1) -> (批量大小, 高度, 宽度, 通道数)
# y_batch.shape: (64, 7) -> (批量大小, 类别数)
print(f"一个批次图像的形状: {X_batch.shape}")
print(f"一个批次标签的形状: {y_batch.shape}")