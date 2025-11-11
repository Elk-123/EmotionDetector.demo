import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# --- A. 数据生成器定义（重复自 data_preparation.py）---
# 确保 DATA_DIR 路径与你的实际情况一致
DATA_DIR = 'data/' 
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7 # 7种情绪

# 归一化，并定义数据增强 (Data Augmentation) 策略
# 简单的数据增强，用于防止模型过拟合
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,        # 随机旋转 10 度
    width_shift_range=0.1,    # 随机水平平移 10%
    height_shift_range=0.1,   # 随机垂直平移 10%
    zoom_range=0.1,           # 随机缩放 10%
    horizontal_flip=True,     # 随机水平翻转
    fill_mode='nearest'
)

# 测试/验证集只需要归一化，不需要增强
test_datagen = ImageDataGenerator(rescale=1./255)

# 加载训练集
train_generator = train_datagen.flow_from_directory(
    directory=DATA_DIR + 'train',
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 加载测试集 (用于验证模型在未见过数据上的表现)
validation_generator = test_datagen.flow_from_directory(
    directory=DATA_DIR + 'test', # 我们用 test 文件夹作为验证集
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# --- B. 模型架构定义：VGG-like CNN ---
def build_vgg_like_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()
    
    # Block 1: 32 Filters
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization()) # 批标准化，加速训练并稳定网络
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) # Dropout，随机关闭神经元，防止过拟合

    # Block 2: 64 Filters
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3: 128 Filters
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 分类器 (Classifier) 部分
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) # 全连接层
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) # 输出层，使用 Softmax 得到概率分布

    return model

# 实例化模型
model = build_vgg_like_model()
model.summary()


# --- C. 模型编译 (Model Compilation) ---
# 选择优化器、损失函数和评估指标
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam 是一种高效的优化器
    loss='categorical_crossentropy', # 适用于多分类任务的损失函数
    metrics=['accuracy']
)

# --- D. 训练回调 (Callbacks) ---
# 用于在训练过程中保存最佳模型和提前停止
checkpoint_filepath = 'models/best_emotion_model.keras' # 保存路径，注意在项目根目录创建 models 文件夹

# 创建 models 文件夹
os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)

# 1. ModelCheckpoint：只保存验证集准确率最高的那个模型权重
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False, # 保存整个模型
    monitor='val_accuracy', # 监控验证集准确率
    mode='max', # 当 val_accuracy 达到最大值时保存
    save_best_only=True
)

# 2. EarlyStopping：如果在一定数量的 epoch 内性能没有改善，则停止训练
early_stopping_callback = EarlyStopping(
    monitor='val_loss', # 监控验证集损失
    patience=10, # 在损失停止下降后，再等待 10 个 epoch
    mode='min',
    restore_best_weights=True # 停止后恢复到性能最佳时的权重
)


# --- E. 模型训练 (Model Training) ---
print("\n--- 开始模型训练 ---")
EPOCHS = 50 # 建议从 50 个 epoch 开始，EarlyStopping 会帮助我们提前停止

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # 每个 epoch 的步数
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)


# --- F. 结果可视化 (可选，用于分析训练过程) ---
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# plot_training_history(history)