import os
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


# Note:
# - Computer Vision: OpenCV + Haar Cascade để phát hiện khuôn mặt từ webcam.
# - Deep Learning: mô hình CNN đã huấn luyện sẵn để phân loại 5 cảm xúc.
MODEL_PATH = "emotion_model_5classes.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAIN_DIR = "face_emotion_split/train"
LABELS_PATH = "emotion_labels.json"

# Load mô hình CNN phân loại 5 lớp cảm xúc:
# Angry, Fear, Happy, Sad, Surprise.
model = tf.keras.models.load_model(MODEL_PATH)

# Quan trọng: đọc metadata nhãn nếu có, nếu chưa có thì suy ra từ thư mục train.
if os.path.isfile(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    emotion_labels = [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]
elif os.path.isdir(TRAIN_DIR):
    emotion_labels = sorted( 
        entry.name for entry in os.scandir(TRAIN_DIR) if entry.is_dir()
    )
else:
    emotion_labels = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# Haar Cascade là thuật toán Computer Vision cổ điển để detect mặt.
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Tối ưu realtime
detect_interval = 5
predict_interval = 2
resize_scale = 0.5
count = 0
faces = []
face_results = []

# Theo dõi riêng từng khuôn mặt gần đúng bằng vị trí tâm khung.
face_histories = {}
face_history_size = 5
track_distance_threshold = 90

color_map = {
    "Angry": (0, 0, 255),
    "Fear": (255, 0, 255),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (255, 255, 0),
}


def preprocess_face(gray_face):
    # Dữ liệu đầu vào cho CNN là ảnh grayscale kích thước 48x48.
    gray_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
    # Chuẩn hóa pixel theo công thức x = x / 255.0 để đưa dữ liệu về [0, 1].
    gray_face = gray_face.astype("float32") / 255.0
    return gray_face


def scale_face_box(face_box, frame_shape, scale):
    x, y, w, h = face_box
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    x = max(0, x)
    y = max(0, y)
    x2 = min(frame_shape[1], x + w)
    y2 = min(frame_shape[0], y + h)
    return x, y, x2, y2


def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def assign_track_id(box, active_track_ids):
    center_x, center_y = get_box_center(box)
    best_track_id = None
    best_distance = None

    for track_id, track in face_histories.items():
        if track_id in active_track_ids:
            continue

        track_x, track_y = track["center"]
        distance = abs(center_x - track_x) + abs(center_y - track_y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_track_id = track_id

    if best_distance is not None and best_distance <= track_distance_threshold:
        return best_track_id

    return f"{center_x}_{center_y}_{time.perf_counter():.4f}"


fps_timer = time.perf_counter()
fps_counter = 0
display_fps = 0.0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    fps_counter += 1

    now = time.perf_counter()
    elapsed = now - fps_timer
    if elapsed >= 1.0:
        display_fps = fps_counter / elapsed
        fps_timer = now
        fps_counter = 0

    small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect khuôn mặt ít thường xuyên hơn vì đây là bước tốn CPU nhất.
    if count % detect_interval == 0:
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(24, 24),
        )

    # Dự đoán theo lô bằng mô hình CNN để giảm lag khi có nhiều khuôn mặt.
    if count % predict_interval == 0 and len(faces) > 0:
        face_results = []
        face_inputs = []
        scaled_boxes = []

        for face_box in faces:
            x1, y1, x2, y2 = scale_face_box(face_box, frame.shape, resize_scale)
            face = gray_frame[y1:y2, x1:x2]
            if face.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            try:
                input_face = preprocess_face(face)
            except cv2.error:
                continue

            face_inputs.append(input_face)
            scaled_boxes.append((x1, y1, x2, y2))

        if face_inputs:
            batch = np.expand_dims(np.array(face_inputs, dtype=np.float32), axis=-1)
            # Softmax ở lớp cuối trả về xác suất cho từng cảm xúc.
            predictions = model(batch, training=False).numpy()
            active_track_ids = set()

            for (x1, y1, x2, y2), pred in zip(scaled_boxes, predictions):
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                raw_emotion = emotion_labels[idx]
                box = (x1, y1, x2, y2)
                track_id = assign_track_id(box, active_track_ids)
                active_track_ids.add(track_id)

                if track_id not in face_histories:
                    face_histories[track_id] = {
                        "center": get_box_center(box),
                        "history": deque(maxlen=face_history_size),
                        "last_seen": count,
                    }

                track = face_histories[track_id]
                track["center"] = get_box_center(box)
                track["last_seen"] = count
                track["history"].append(raw_emotion)
                smooth_emotion = max(set(track["history"]), key=track["history"].count)

                face_results.append((x1, y1, x2, y2, smooth_emotion, conf, raw_emotion))

            stale_track_ids = [
                track_id
                for track_id, track in face_histories.items()
                if count - track["last_seen"] > detect_interval * 4
            ]
            for track_id in stale_track_ids:
                del face_histories[track_id]

    elif len(faces) == 0:
        face_results = []

    for x1, y1, x2, y2, emotion, conf, raw_emotion in face_results:
        color = color_map.get(emotion, (0, 255, 0))
        label_text = f"{emotion} ({conf * 100:.1f}%)"

        if emotion != raw_emotion:
            label_text = f"{emotion} ~ {raw_emotion} ({conf * 100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label_text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    cv2.putText(
        frame,
        f"FPS: {display_fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Faces: {len(face_results)} | Detect every {detect_interval}f | Predict every {predict_interval}f",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Press Q to exit",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Emotion AI FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
