# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# ================= 配置区域 =================
IMG_SIZE = 224       
NUM_CLASSES = 7
BATCH_SIZE = 32      
EPOCHS = 50
LEARNING_RATE = 1e-4

# 路径配置
BASE_DIR = 'archive'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'val')   # 用于训练中的验证
TEST_DIR  = os.path.join(BASE_DIR, 'test')  # 用于最终测试

MODEL_DIR = 'models'
MODEL_NAME = 'resnet50_rafdb_model.keras' 

os.makedirs(MODEL_DIR, exist_ok=True)
# ===========================================

# 1. 数据生成器 Setup
print("INFO: Setting up Data Generators...")

# 训练集：增强
train_datagen = ImageDataGenerator(
    rescale=1./255,             
    rotation_range=20,          
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 验证集和测试集：仅归一化
# (注意：验证集和测试集都不应该进行数据增强，只需要 rescale)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# --- 生成器 A: 训练 ---
print(f"INFO: Loading Training data from {TRAIN_DIR}...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',           
    class_mode='categorical',
    shuffle=True
)

# --- 生成器 B: 验证 (用于 fit 过程中的 validation_data) ---
print(f"INFO: Loading Validation data from {VAL_DIR}...")
validation_generator = valid_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',           
    class_mode='categorical',
    shuffle=False # 验证集不需要打乱
)

# --- 生成器 C: 测试 (用于训练后的最终评估) ---
print(f"INFO: Loading Test data from {TEST_DIR}...")
test_generator = valid_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',           
    class_mode='categorical',
    shuffle=False # 测试集也不需要打乱
)

# 2. 定义 ResNet50 模型
def create_resnet50_model(input_shape, num_classes):
    input_tensor = tf.keras.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor 
    )

    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)
    return model

model = create_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)

# 3. 编译
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 回调函数
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, MODEL_NAME),
        monitor='val_accuracy', # 监控 val 集的准确率
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# 5. 开始训练 (Train vs Val)
print("\nINFO: Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator, # 使用 Val 集进行过程监控
    callbacks=callbacks
)

# 6. 最终测试 (Evaluation on Test Set)
print("\nINFO: Training completed. Evaluating on Test Set...")
# 加载刚才保存的最好模型（因为 EarlyStopping 可能会回滚，或者 best model 是几轮之前的）
best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME))

test_loss, test_acc = best_model.evaluate(test_generator)
print(f"\n==========================================")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"==========================================\n")