这是一个非常好的习惯！在完成一个项目阶段后进行总结和文档化，能极大地提高项目的可维护性和专业性。

下面是为你项目量身定做的 **项目总结和技术文档**，文件名为 `Guide.md`。它包含了我们已完成的步骤、当前状态，以及后续的规划。

---

## 📄 Guide.md：智能情绪识别器项目文档

### 🌟 项目名称：智能情绪识别器 (Emotion Detector)

**目标：** 构建一个基于卷积神经网络 (CNN) 的计算机视觉系统，能够实时识别图像或视频流中的人脸情绪（7 种类别）。

**当前状态：** **阶段 I (数据准备) 已完成。阶段 II (模型训练) 正在进行中。**

---

## 🚀 阶段 I：数据准备与预处理 (已完成)

**目标：** 获取、探索并预处理 FER-2013 数据集，使其适用于深度学习模型训练。

### 1. 数据集

* **数据集：** FER-2013 (按文件夹划分版本)
* **文件结构：** 按类别组织的文件夹结构（`data/train/EmotionName` 和 `data/test/EmotionName`）
* **情绪类别 (7种)：** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* **数据总量：** 训练集约 28,709 张，测试集约 7,178 张。

### 2. 技术要点与实现 (`src/data_preparation.py` / `src/model_training.py`)

| 技术点 | 描述 |
| :--- | :--- |
| **数据加载** | 使用 Keras 的 `ImageDataGenerator` 和 `flow_from_directory`，直接从文件夹加载数据。 |
| **图像尺寸** | 图像统一缩放为模型所需的 $48 \times 48$ 像素。 |
| **通道模式** | 统一采用 `color_mode='grayscale'` (单通道)。 |
| **标签编码** | 采用 `class_mode='categorical'` (独热编码)，适合多分类任务。 |
| **数据归一化** | 通过 `rescale=1./255` 将像素值从 [0, 255] 缩放到 [0, 1]。 |
| **数据增强** | 训练集启用了 `rotation_range`, `width_shift_range`, `horizontal_flip` 等操作，以增加数据多样性，防止模型过拟合。 |

---

## 🧠 阶段 II：模型构建与训练 (进行中)

**目标：** 设计并训练一个 VGG-like CNN 模型，以最大化情绪分类的准确率。

### 1. 模型架构 (`src/model_training.py` - `build_vgg_like_model`)

* **类型：** VGG-like 卷积神经网络 (CNN)。
* **结构特点：** 采用经典的 Conv-Conv-MaxPool 块堆叠，逐层提取更复杂的特征。
    * **深度：** 3 个主要的卷积块 (32, 64, 128 个滤波器)。
    * **正则化：** 广泛使用了 **`BatchNormalization`** (加速收敛) 和 **`Dropout`** (防止过拟合)。
    * **输出：** 7 个神经元的全连接层，激活函数为 **`softmax`**。
* **总参数量：** 约 **265 万** (10.12 MB)。

### 2. 训练配置

| 配置项 | 值 / 描述 |
| :--- | :--- |
| **优化器** | `Adam(learning_rate=0.001)` |
| **损失函数** | `categorical_crossentropy` (多分类标准损失) |
| **指标** | `accuracy` |
| **批次大小 (Batch Size)** | `64` |
| **周期数 (Epochs)** | `50` (由回调机制控制实际运行数量) |
| **回调函数** | **`ModelCheckpoint`**: 监控 `val_accuracy`，只保存最佳模型。**`EarlyStopping`**: 监控 `val_loss`，连续 10 次不改进则停止训练。 |

### 3. 当前进度与待办事项

* 模型正在训练中，终端显示 Epoch 进度。
* **待办 (训练完成时)：**
    * **结果分析：** 检查 `val_loss` 和 `val_accuracy` 趋势，确定模型是否收敛。
    * **模型保存：** 最佳模型已自动保存在 `models/best_emotion_model.keras`。

---

## 🎯 阶段 III：模型评估与实时部署 (未完成 - 规划中)

**目标：** 评估训练结果，并将模型集成到实时视频流应用中。

### 1. 模型评估

* **文件：** `src/model_evaluation.py` (待创建)
* **功能：**
    * 加载 `best_emotion_model.keras`。
    * 在测试集上计算最终准确率、召回率、F1 分数。
    * 生成并可视化 **混淆矩阵 (Confusion Matrix)**，特别是分析模型在 **Disgust** 等稀有类别上的表现。

### 2. 实时识别部署

* **文件：** `src/realtime_detector.py` (代码骨架已提供)
* **所需库：** `opencv-python` (需安装)
* **所需文件：** `haarcascade_frontalface_default.xml` (需下载并放在项目根目录)
* **实现步骤：**
    1.  初始化摄像头 (`cv2.VideoCapture`)。
    2.  在视频帧中，使用 **Haar Cascade 分类器** 快速检测人脸。
    3.  提取人脸区域 (ROI)，预处理（灰度、缩放 $48 \times 48$、归一化）。
    4.  使用加载的 Keras 模型预测情绪。
    5.  使用 OpenCV 在实时视频流上绘制人脸框和预测的情绪标签。

---

### 💡 下一步建议 (基于你的训练进度)：

请在训练完成后，关注以下输出：

1.  **最终准确率：** `val_accuracy` 的最高值。
2.  **文件确认：** 确认 `models/best_emotion_model.keras` 文件已生成。

接下来，我们将进入 **阶段 III** 的实际操作，从**模型评估**开始。