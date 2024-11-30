import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore

# YOLOモデルの読み込み
model = YOLO("yolov8x")  # 学習済みモデルのパスを指定

# IPカメラのURL
ip_camera_url = 0

# 検出対象のクラスID
HAND_CLASS_ID = 0  # 手のクラスID
TARGET_CLASS_IDS = [39, 40, 41]  # 物体のクラスIDを配列で登録（例: 39, 40, 41）

# 動きの追跡用
def calculate_movement(positions, new_position):
    positions.append(new_position)
    if len(positions) > 2:
        positions.pop(0)
    if len(positions) == 2:
        return np.array(positions[1]) - np.array(positions[0])
    return np.array([0, 0])


cap = cv2.VideoCapture(ip_camera_url)

hand_positions = []
object_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    hands = []
    objects = []

    # 検出された物体を分類
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        if class_id == HAND_CLASS_ID:  # 手のクラスID
            hands.append(center)
        elif class_id in TARGET_CLASS_IDS:  # 登録された物体のクラスID
            objects.append(center)

    # 動きを計算
    hand_movement = np.array([0, 0])
    object_movement = np.array([0, 0])
    
    for hand in hands:
        hand_movement = calculate_movement(hand_positions, hand)
    for obj in objects:
        object_movement = calculate_movement(object_positions, obj)

    # 同じ方向・速度の判定
    holding = False
    if len(hands) > 0 and len(objects) > 0:
        if np.linalg.norm(hand_movement - object_movement) < 5:  # 速度の閾値
            holding = True

    # 結果を表示
    for hand in hands:
        cv2.circle(frame, (int(hand[0]), int(hand[1])), 5, (255, 0, 0), -1)  # 手を青い円で表示
    for obj in objects:
        cv2.circle(frame, (int(obj[0]), int(obj[1])), 5, (0, 255, 0), -1)  # 物体を緑の円で表示
    if holding:
        cv2.putText(frame, "Holding Object", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
