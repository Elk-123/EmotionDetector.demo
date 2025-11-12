# 导入必要的库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os

# 定义常量和路径
IMAGE_SIZE = 48
BATCH_SIZE = 64
TEST_DIR = 'data/test'
MODEL_PATH = 'models/best_emotion_model.keras'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- 辅助函数：绘制混淆矩阵 ---
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图。
    参数：
        cm: 混淆矩阵 (numpy array)
        classes: 情绪类别标签 (list)
        title: 图表标题 (string)
        cmap: 颜色映射 (matplotlib colormap)
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 打印归一化后的数字
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label (真实标签)')
    plt.xlabel('Predicted label (预测标签)')
    plt.show()

# --- 主函数 ---
def evaluate_model():
    # 1. 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}。请确认模型训练并保存成功。")
        return
    if not os.path.exists(TEST_DIR):
        print(f"错误: 找不到测试数据集目录 {TEST_DIR}。请确认数据准备正确。")
        return

    print("--- 1. 加载模型 ---")
    try:
        # 加载最佳模型
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"模型 {MODEL_PATH} 加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 准备测试数据生成器
    print("\n--- 2. 准备测试数据 ---")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False  # 评估时必须关闭 shuffle 以确保标签和预测顺序一致
    )
    
    # 3. 评估模型
    print("\n--- 3. 评估模型在测试集上的性能 ---")
    loss, acc = model.evaluate(test_generator)
    print(f"\n最终测试集损失 (Test Loss): {loss:.4f}")
    print(f"最终测试集准确率 (Test Accuracy): {acc:.4f}")

    # 4. 获取预测结果
    print("\n--- 4. 生成预测结果和混淆矩阵 ---")
    
    # 获取测试集的步数
    steps = test_generator.samples // BATCH_SIZE + (test_generator.samples % BATCH_SIZE != 0)
    
    # 使用模型进行预测
    Y_pred = model.predict(test_generator, steps=steps)
    
    # 将独热编码的预测结果转换为类别索引
    y_pred = np.argmax(Y_pred, axis=1)
    
    # 获取真实的类别索引
    y_true = test_generator.classes

    # 确认标签顺序与生成器的一致
    labels = list(test_generator.class_indices.keys())
    
    # 5. 打印分类报告
    print("\n--- 5. 详细分类报告 ---")
    # 报告包含 准确率(precision), 召回率(recall), F1分数(f1-score)
    print(classification_report(y_true, y_pred, target_names=labels))

    # 6. 计算和绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=labels, title='Normalized Confusion Matrix (混淆矩阵)')

# 执行评估
if __name__ == '__main__':
    # 确保TensorFlow不会因为GPU内存不足而崩溃
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 允许内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 必须在程序启动时设置
            print(e)
            
    evaluate_model()