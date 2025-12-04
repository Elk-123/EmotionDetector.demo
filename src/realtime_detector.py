# --- START OF FILE src/realtime_detector.py ---

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import deque

# ================= 配置区 =================
EMOTION_MODEL_PATH = 'models/resnet50_emotion_model.keras'
YOLO_MODEL_NAME = 'yolov8n-face.pt'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48

# --- 核心优化配置 ---
# 1. 降低检测门槛：解决“识别不出来”的问题
YOLO_CONF_THRESHOLD = 0.4  

# 2. 记忆缓冲：解决“ID乱跳”的问题
# 允许人脸丢失多少帧后才注销 ID？(设为 10 帧，约 0.3 秒)
MAX_DISAPPEARED = 10 

# 3. 情绪平滑窗口
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
        # 存储: { id: {'box': [x1,y1,x2,y2], 'disappeared': 0, 'probs': deque, 'label': '..'} }
        self.tracked_objects = {} 
        self.next_object_id = 0

    def register(self, box):
        """注册新 ID"""
        self.tracked_objects[self.next_object_id] = {
            'box': box,
            'disappeared': 0,
            'probs': deque(maxlen=SMOOTH_WINDOW), # 情绪平滑队列
            'current_label': 'Detecting...',
            'current_conf': 0.0
        }
        self.next_object_id += 1

    def deregister(self, object_id):
        """注销 ID"""
        del self.tracked_objects[object_id]

    def update(self, rects):
        """
        核心追踪逻辑：基于 IoU 的匹配
        rects: 当前帧所有检测到的框
        """
        # 1. 如果没有正在追踪的对象，全部注册为新对象
        if len(self.tracked_objects) == 0:
            for rect in rects:
                self.register(rect)
            return self.tracked_objects

        # 2. 准备数据
        object_ids = list(self.tracked_objects.keys())
        object_values = list(self.tracked_objects.values())
        tracked_boxes = [obj['box'] for obj in object_values]

        # 3. 如果当前帧没有检测到人脸，所有人 disappeared + 1
        if len(rects) == 0:
            for object_id in object_ids:
                self.tracked_objects[object_id]['disappeared'] += 1
                if self.tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED:
                    self.deregister(object_id)
            return self.tracked_objects

        # 4. 计算 IoU 矩阵 (旧框 vs 新框)
        # 这是一个简单的贪婪匹配逻辑
        used_rows = set() # 已匹配的旧 ID 索引
        used_cols = set() # 已匹配的新框索引

        # 计算所有可能的 IoU
        matches = []
        for i, old_box in enumerate(tracked_boxes):
            for j, new_box in enumerate(rects):
                iou = calculate_iou(old_box, new_box)
                if iou > 0.3: # 只有重叠率 > 30% 才认为是同一个
                    matches.append((iou, i, j))
        
        # 按 IoU 从大到小排序，优先匹配重叠度最高的
        matches.sort(key=lambda x: x[0], reverse=True)

        for iou, row, col in matches:
            if row in used_rows or col in used_cols:
                continue

            # 匹配成功：更新框，重置消失计数
            object_id = object_ids[row]
            self.tracked_objects[object_id]['box'] = rects[col]
            self.tracked_objects[object_id]['disappeared'] = 0
            
            used_rows.add(row)
            used_cols.add(col)

        # 5. 处理未匹配的旧 ID (认为暂时消失)
        for i in range(len(object_ids)):
            if i not in used_rows:
                object_id = object_ids[i]
                self.tracked_objects[object_id]['disappeared'] += 1
                if self.tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED:
                    self.deregister(object_id)

        # 6. 处理未匹配的新框 (注册为新 ID)
        for i in range(len(rects)):
            if i not in used_cols:
                self.register(rects[i])

        return self.tracked_objects

# ================= 初始化 =================
print("--- 初始化增强版系统 ---")
try:
    # 尝试加载增强模型，如果没有就加载普通模型
    try:
        emotion_model = load_model('models/best_emotion_model_enhanced.keras')
        print("✅ 加载了增强版情绪模型 (Enhanced)")
    except:
        emotion_model = load_model(EMOTION_MODEL_PATH)
        print("⚠️ 加载了普通版情绪模型 (Standard)")
        
    face_model = YOLO(YOLO_MODEL_NAME)
    print("✅ YOLOv8 检测系统就绪 (低阈值模式)")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    exit()

tracker = AdvancedFaceTracker()
cap = cv2.VideoCapture(0)

print(f"\n🚀 系统运行中 | 检测阈值: {YOLO_CONF_THRESHOLD} | ID记忆: {MAX_DISAPPEARED}帧")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # 镜像
    
    # 1. YOLO 检测 (使用更低的 conf 阈值)
    results = face_model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
    
    rects = []
    if results[0].boxes:
        # 只取坐标，转为 int 列表
        boxes = results[0].boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            rects.append([x1, y1, x2, y2])

    # 2. 追踪器更新 (核心)
    # 这一步返回的是所有"活着的" ID，包括短暂消失但还在记忆里的
    objects = tracker.update(rects)

    # 3. 遍历处理
    for obj_id, data in objects.items():
        # 如果这个 ID 当前处于"消失中"状态 (disappeared > 0)，就不画框，也不预测
        if data['disappeared'] > 0:
            continue

        x1, y1, x2, y2 = data['box']
        
        # --- 预处理 (增加 Padding，解决大头照问题) ---
        h_img, w_img, _ = frame.shape
        pad = int((y2 - y1) * 0.2) # 20% Padding
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w_img, x2 + pad)
        y2_p = min(h_img, y2 + pad)
        
        face_roi = frame[y1_p:y2_p, x1_p:x2_p]
        
        if face_roi.size > 0:
            # --- 情绪预测 ---
            try:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                normalized = resized.astype('float32') / 255.0
                input_data = np.expand_dims(np.expand_dims(normalized, -1), 0)

                # 预测
                preds = emotion_model.predict(input_data, verbose=0)[0]
                
                # --- 平滑处理 ---
                data['probs'].append(preds)
                avg_preds = np.mean(data['probs'], axis=0)
                
                idx = np.argmax(avg_preds)
                data['current_label'] = EMOTIONS[idx]
                data['current_conf'] = avg_preds[idx]

            except Exception:
                pass

        # --- 绘制 UI ---
        label = data['current_label']
        conf = data['current_conf']
        
        # 根据情绪变色
        color = (0, 255, 0) # 默认绿
        if label in ['Angry', 'Disgust', 'Fear', 'Sad']: 
            color = (0, 0, 255) # 红色
        elif label == 'Happy':
            color = (0, 255, 255) # 黄色
            
        # 绘制人脸框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制背景条让文字更清楚
        info_text = f"ID:{obj_id} {label} {int(conf*100)}%"
        t_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), color, -1)
        
        # 绘制文字
        cv2.putText(frame, info_text, (x1, y1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow('Pro Emotion Detector', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()