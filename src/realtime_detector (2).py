import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import deque
import os

# ================= 基础配置（无需手动修改）=================
EMOTION_MODEL_PATH = 'models/resnet50_rafdb_model.keras'
YOLO_MODEL_NAME = 'yolov8n.pt'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
tf.keras.config.enable_unsafe_deserialization()

# --- 核心优化配置 ---
YOLO_CONF_THRESHOLD = 0.4  
MAX_DISAPPEARED = 10 
SMOOTH_WINDOW = 8 

# ==========================================

def calculate_iou(boxA, boxB):
    """计算两个矩形框的重叠率 (Intersection over Union)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class AdvancedFaceTracker:
    def __init__(self):
        self.tracked_objects = {} 
        self.next_object_id = 0

    def register(self, box):
        """注册新 ID"""
        self.tracked_objects[self.next_object_id] = {
            'box': box,
            'disappeared': 0,
            'probs': deque(maxlen=SMOOTH_WINDOW),
            'current_label': 'Detecting...',
            'current_conf': 0.0
        }
        self.next_object_id += 1

    def deregister(self, object_id):
        """注销 ID"""
        del self.tracked_objects[object_id]

    def update(self, rects):
        """核心追踪逻辑：基于 IoU 的匹配"""
        if len(self.tracked_objects) == 0:
            for rect in rects:
                self.register(rect)
            return self.tracked_objects

        object_ids = list(self.tracked_objects.keys())
        object_values = list(self.tracked_objects.values())
        tracked_boxes = [obj['box'] for obj in object_values]

        if len(rects) == 0:
            for object_id in object_ids:
                self.tracked_objects[object_id]['disappeared'] += 1
                if self.tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED:
                    self.deregister(object_id)
            return self.tracked_objects

        used_rows = set()
        used_cols = set()
        matches = []

        for i, old_box in enumerate(tracked_boxes):
            for j, new_box in enumerate(rects):
                iou = calculate_iou(old_box, new_box)
                if iou > 0.3:
                    matches.append((iou, i, j))
        
        matches.sort(key=lambda x: x[0], reverse=True)

        for iou, row, col in matches:
            if row in used_rows or col in used_cols:
                continue
            object_id = object_ids[row]
            self.tracked_objects[object_id]['box'] = rects[col]
            self.tracked_objects[object_id]['disappeared'] = 0
            used_rows.add(row)
            used_cols.add(col)

        for i in range(len(object_ids)):
            if i not in used_rows:
                object_id = object_ids[i]
                self.tracked_objects[object_id]['disappeared'] += 1
                if self.tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED:
                    self.deregister(object_id)

        for i in range(len(rects)):
            if i not in used_cols:
                self.register(rects[i])

        return self.tracked_objects

# ================= 智能初始化（自动适配模型输入）=================
print("--- 初始化增强版系统 ---")
tf.keras.config.enable_unsafe_deserialization()

try:
    # 检查模型文件
    if not os.path.exists(EMOTION_MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {EMOTION_MODEL_PATH}")
    
    # 加载模型并自动识别输入配置
    emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)
    model_input_shape = emotion_model.input_shape
    print(f"✅ 成功加载 ResNet50 模型")
    print(f"📌 模型期望输入形状: {model_input_shape}")
    
    # 自动提取输入参数（无需手动修改）
    IMG_SIZE = model_input_shape[1]  # 自动获取尺寸（48或120）
    INPUT_CHANNELS = model_input_shape[3]  # 自动获取通道数（1或3）
    
    print(f"🔍 自动适配配置:")
    print(f"  - 输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - 通道数: {INPUT_CHANNELS} {'(灰度图)' if INPUT_CHANNELS == 1 else '(RGB图)'}")
    
    # 加载 YOLO 人脸检测模型
    face_model = YOLO(YOLO_MODEL_NAME)
    print("✅ YOLOv8 人脸检测系统就绪 (低阈值模式)")
    
except Exception as e:
    print(f"❌ 初始化失败: {str(e)}")
    exit()

# 初始化追踪器和摄像头
tracker = AdvancedFaceTracker()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头，请检查设备连接")
    exit()

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"\n🚀 系统运行中 | 检测阈值: {YOLO_CONF_THRESHOLD} | ID记忆: {MAX_DISAPPEARED}帧")
print(f"📊 适配后配置: {IMG_SIZE}x{IMG_SIZE} {'(灰度图)' if INPUT_CHANNELS == 1 else '(RGB图)'}")
print("按 'q' 或 ESC 退出")

# ================= 主循环（智能预处理）=================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取摄像头画面，正在重试...")
        continue
    frame = cv2.flip(frame, 1)  # 镜像翻转
    
    # 1. YOLO 人脸检测（只检测人体，类别0）
    results = face_model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD, classes=[0])
    rects = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # 过滤过小的框（自适应模型输入尺寸）
            min_size = int(IMG_SIZE * 0.8)
            if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                rects.append([x1, y1, x2, y2])

    # 2. 人脸追踪
    objects = tracker.update(rects)

    # 3. 情绪识别（智能预处理，自动适配通道数）
    for obj_id, data in objects.items():
        if data['disappeared'] > 0:
            continue

        x1, y1, x2, y2 = data['box']
        
        # 预处理：自适应Padding
        h_img, w_img, _ = frame.shape
        pad_ratio = 0.15 if IMG_SIZE <= 64 else 0.2  # 小尺寸输入用小Padding
        pad = int((y2 - y1) * pad_ratio)
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        
        # 提取人脸ROI
        face_roi = frame[y1_p:y2_p, x1_p:x2_p]
        
        if face_roi.size > 0:
            try:
                # 智能预处理（自动适配模型输入）
                if INPUT_CHANNELS == 1:
                    # 模型要求灰度图（1通道）
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    normalized = resized.astype('float32') / 255.0
                    # 增加通道维度和批次维度
                    input_data = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
                else:
                    # 模型要求RGB图（3通道）
                    resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    # 确保通道数正确
                    if len(resized.shape) == 2:
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    elif resized.shape[2] == 4:
                        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
                    normalized = resized.astype('float32') / 255.0
                    # 增加批次维度
                    input_data = np.expand_dims(normalized, axis=0)
                
                # 验证输入形状（首次运行可开启调试）
                # print(f"📥 输入形状: {input_data.shape}")
                
                # 情绪预测
                preds = emotion_model.predict(input_data, verbose=0)[0]
                
                # 情绪平滑
                data['probs'].append(preds)
                min_smooth_frames = 2 if IMG_SIZE <= 64 else 3
                avg_preds = np.mean(data['probs'], axis=0) if len(data['probs']) >= min_smooth_frames else preds
                
                # 获取最终情绪和置信度
                emotion_idx = np.argmax(avg_preds)
                data['current_label'] = EMOTIONS[emotion_idx]
                data['current_conf'] = avg_preds[emotion_idx]

            except Exception as e:
                print(f"⚠️ 预测失败 (ID:{obj_id}): {str(e)[:60]}")
                pass

        # 4. 绘制UI界面
        label = data['current_label']
        conf = data['current_conf']
        
        # 情绪颜色映射
        color_map = {
            'Angry': (0, 0, 255),      # 红色
            'Disgust': (128, 0, 128),  # 紫色
            'Fear': (255, 0, 255),     # 洋红
            'Happy': (0, 255, 255),    # 黄色
            'Sad': (255, 0, 0),        # 蓝色
            'Surprise': (255, 165, 0), # 橙色
            'Neutral': (255, 255, 0),  # 青色
            'Detecting...': (128, 128, 128) # 灰色
        }
        color = color_map.get(label, (0, 255, 0))
        
        # 绘制人脸框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        info_text = f"ID:{obj_id} {label} {int(conf*100)}%"
        t_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bg_x2 = min(x1 + t_size[0], w_img - 5)
        cv2.rectangle(frame, (x1, y1 - 25), (bg_x2, y1), color, -1)
        
        # 绘制文字
        cv2.putText(frame, info_text, (x1, y1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 显示系统状态
    active_faces = len([obj for obj in objects.values() if obj['disappeared'] == 0])
    mode_text = "Grayscale" if INPUT_CHANNELS == 1 else "RGB"
    stats_text = f"Active Faces: {active_faces} | {IMG_SIZE}x{IMG_SIZE}-{mode_text}"
    cv2.putText(frame, stats_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 显示画面
    cv2.imshow('Pro Emotion Detector (Auto-Adaptive)', frame)
    
    # 退出逻辑
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("🛑 正在退出系统...")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("👋 系统已安全退出")