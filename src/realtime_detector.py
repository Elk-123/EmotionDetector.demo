# 导入必要的库
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# --- 配置常量和路径 ---
IMAGE_SIZE = 48
MODEL_PATH = 'models/best_emotion_model.keras'
# Haar Cascade分类器的路径，请确保它位于项目根目录
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml' 

# 7种情绪标签，必须与模型训练时的顺序一致
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

# 用于在视频帧上绘制预测结果的颜色
COLOR_MAP = {
    'Happy': (0, 255, 0),    # 绿色
    'Sad': (255, 0, 0),      # 蓝色
    'Angry': (0, 0, 255),    # 红色
    'Fear': (0, 255, 255),   # 黄色
    'Surprise': (255, 255, 0), # 青色
    'Neutral': (128, 128, 128),# 灰色
    'Disgust': (0, 165, 255) # 橙色
}

# --- 实时识别主函数 ---
def real_time_emotion_detector():
    print("--- 1. 启动情绪识别器 ---")

    # 1. 检查模型和Haar Cascade文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}。请确认模型已训练并保存。")
        return
    if not os.path.exists(FACE_CASCADE_PATH):
        print(f"错误: 找不到人脸检测文件 {FACE_CASCADE_PATH}。")
        print("请将 'haarcascade_frontalface_default.xml' 放入项目根目录。")
        return

    # 2. 加载模型和人脸检测器
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"模型 {MODEL_PATH} 加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    try:
        face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        print("人脸检测器加载成功。")
    except Exception as e:
        print(f"加载人脸检测器失败: {e}")
        return

    # 3. 初始化摄像头
    # 尝试打开第一个摄像头 (索引 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。请检查摄像头连接或权限设置。")
        return

    print("--- 2. 实时预测中 (按 'q' 退出) ---")
    
    # 计时器用于控制预测频率
    prediction_interval = 5  # 每 5 帧进行一次预测
    frame_count = 0
    current_emotion = "Starting..."

    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        if not ret:
            print("警告: 无法从摄像头读取帧。")
            break

        # 图像预处理
        # 实时识别通常在灰度图上进行，以减少计算量
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 每隔几帧才进行一次人脸检测和情绪预测，以提高帧率
        if frame_count % prediction_interval == 0:
            # 4. 人脸检测 (使用 Haar Cascade)
            # scaleFactor=1.1, minNeighbors=5 是常用参数
            faces = face_detector.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30), 
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 如果检测到人脸，只对最大的一张人脸进行情绪识别
            if len(faces) > 0:
                # 找到最大的人脸
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # 提取人脸区域 (Region of Interest, ROI)
                roi_gray = gray_frame[y:y + h, x:x + w]
                
                # 5. 图像预处理和模型输入准备
                # 缩放至模型要求的 48x48 尺寸
                cropped_img = cv2.resize(roi_gray, (IMAGE_SIZE, IMAGE_SIZE))
                # 归一化 (0-1范围)
                normalized_img = cropped_img / 255.0
                # 调整形状为 (1, 48, 48, 1) 以匹配模型的输入要求 (Batch, Height, Width, Channel)
                input_tensor = np.expand_dims(np.expand_dims(normalized_img, -1), 0)
                
                # 6. 情绪预测
                predictions = model.predict(input_tensor, verbose=0)[0]
                # 获取概率最大的情绪索引
                emotion_index = np.argmax(predictions)
                # 获取情绪标签
                current_emotion = EMOTION_LABELS[emotion_index]
                # 获取最高概率
                confidence = predictions[emotion_index] * 100
                
                # 7. 在视频帧上绘制结果
                # 选择颜色
                color = COLOR_MAP.get(current_emotion, (255, 255, 255)) 
                
                # 绘制人脸矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 绘制情绪标签和置信度
                text = f"{current_emotion}: {confidence:.1f}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # 重置计数器
            frame_count = 0
            
        # 无论是否预测，都需要递增计数器
        frame_count += 1
        
        # 在整个窗口上显示当前预测的情绪，即使没有在这一帧进行预测
        cv2.putText(frame, current_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 8. 显示最终画面
        cv2.imshow('Real-time Emotion Detector', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    print("\n情绪识别器已关闭。")

if __name__ == '__main__':
    real_time_emotion_detector()