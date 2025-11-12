import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. 配置和路径
# -----------------------------------------------------------
# 模型文件路径
MODEL_PATH = 'models/best_emotion_model.keras'
# Haar Cascade 人脸检测器文件路径
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# 情绪类别列表 (与训练时保持一致)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48  # 模型输入图像尺寸 (48x48)

# --- 实时性能优化变量 ---
# SKIP_FRAME = 3: 每隔 3 帧进行一次模型预测，提高流畅度和帧率
SKIP_FRAME = 3
frame_count = 0 

# 上一次的预测结果，用于在跳过的帧中显示，保持识别框的稳定
last_prediction = "Detecting..."
last_confidence = 0.0

# 2. 初始化模型和分类器
# -----------------------------------------------------------
try:
    # 尝试加载训练好的 Keras 模型
    model = load_model(MODEL_PATH)
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
    
    # 尝试加载人脸分类器
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"ERROR: Could not load Haar Cascade file at {FACE_CASCADE_PATH}")

except Exception as e:
    print(f"FATAL ERROR during initialization: {e}")
    print("请检查 MODEL_PATH, FACE_CASCADE_PATH 是否正确，并确保已安装依赖 (tensorflow, opencv-python, h5py)。")
    exit()

# 3. 启动摄像头
# -----------------------------------------------------------
# 0 表示默认摄像头，如果检测不到可以尝试 1 或其他编号
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open video stream or file.")
    exit()

print("\nINFO: 实时情绪识别器正在运行。按 'q' 键退出。")

# 4. 实时处理循环
# -----------------------------------------------------------
while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    # 翻转图像，使其更自然（镜像效果）
    frame = cv2.flip(frame, 1)
    
    # 将帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 Haar Cascade 检测人脸
    # 注意：移除了不兼容的 flags=cv2.IMAGENET_FLAGS 参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.IMAGENET_FLAGS # 移除此行以避免AttributeError
    )

    
    # 循环处理所有检测到的人脸
    for (x, y, w, h) in faces:
        
        # 优化策略 C.2：增加边界填充 (Padding)
        # 增加 10 像素的边界，避免人脸裁剪过紧
        padding = 10 
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # 裁剪人脸区域 (ROI) - 使用填充后的坐标
        roi_gray = gray[y1:y2, x1:x2]

        
        # --- 性能优化：跳帧预测 ---
        if frame_count % SKIP_FRAME == 0:
            
            # 优化策略 C.1：直方图均衡化（增强对比度）
            if roi_gray.size > 0: # 确保裁剪区域不为空
                roi_gray_processed = cv2.equalizeHist(roi_gray)
            else:
                continue

            # 5. 预处理和模型预测（只在关键帧执行）
            
            # 调整尺寸到模型输入大小 (48x48)
            cropped_img = cv2.resize(roi_gray_processed, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # 归一化到 [0, 1] 范围
            cropped_img = cropped_img.astype('float32') / 255.0
            
            # Keras模型需要 (batch_size, height, width, channels) 格式
            # 这里是 (1, 48, 48, 1)
            processed_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
            
            # 预测
            predictions = model.predict(processed_img, verbose=0)[0]
            
            # 获取预测结果和置信度
            predicted_class_index = np.argmax(predictions)
            last_prediction = EMOTIONS[predicted_class_index]
            last_confidence = predictions[predicted_class_index]
        
        # 6. 绘制结果（每帧都绘制）
        
        # 绘制人脸框 (使用原始检测坐标，以便准确标记)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 格式化输出文本
        text = f"{last_prediction}: {last_confidence*100:.2f}%"
        
        # 在人脸框上方显示预测结果
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 更新帧计数器
    frame_count += 1
    
    # 7. 显示和退出
    # -----------------------------------------------------------
    # 显示处理后的帧
    cv2.imshow('Real-time Emotion Detector (Optimized)', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. 清理资源
# -----------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("INFO: 程序退出，资源已释放。")