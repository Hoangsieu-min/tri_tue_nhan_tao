import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Kiểm tra webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("LỖI: Không tìm thấy webcam! Hãy kiểm tra:")
    print("  - Laptop: webcam có bị tắt trong Device Manager không?")
    print("  - PC: đã cắm webcam USB chưa?")
    print("  - Thử đổi số 0 thành 1 hoặc 2 trong cv2.VideoCapture(1)")
    exit(1)
else:
    print("Đã tìm thấy webcam. Đang khởi tạo...")

# 2. Load mô hình đã huấn luyện (5 cảm xúc)
try:
    model = load_model('emotion_model_5classes.h5')
    print("Đã load mô hình thành công.")
except Exception as e:
    print("LỖI: Không thể load mô hình 'emotion_model_5classes.h5'")
    print("   Bạn cần chạy train_model_pro.py trước để tạo mô hình.")
    exit(1)

# 3. Nhãn cảm xúc (theo thứ tự class_indices khi train)
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

# 4. Haar Cascade phát hiện mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Nhấn 'q' để thoát. Bắt đầu nhận diện...")

# 5. Vòng lặp realtime
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình từ webcam. Thoát.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48,48))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)   # shape (1,48,48,1)

        preds = model.predict(roi, verbose=0)
        idx = np.argmax(preds)
        label = emotion_labels[idx]
        confidence = np.max(preds)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
print("Đã thoát chương trình.")