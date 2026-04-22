import os
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt

# Note tổng quan:
# - Dữ liệu: ảnh khuôn mặt grayscale 48x48, chia thành 5 lớp cảm xúc.
# - Mô hình: CNN xây bằng TensorFlow/Keras.
# - Công thức chính: chuẩn hóa x = x / 255.0, đầu ra Softmax, loss là Categorical Crossentropy.

# ==================== KIỂM TRA THƯ MỤC ====================
train_dir = 'face_emotion_split/train'
test_dir = 'face_emotion_split/test'
img_size = (48, 48)
batch_size = 64
epochs = 30

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

# ==================== DATA AUGMENTATION ====================
# rescale=1./255 tương đương công thức chuẩn hóa pixel: x' = x / 255.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# ==================== GENERATORS ====================
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True           # ✅ xáo trộn dữ liệu mỗi epoch
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False          # ✅ không cần xáo trộn khi đánh giá
)

print("Các lớp (thứ tự):", train_generator.class_indices)

# ==================== MÔ HÌNH ====================
# Kiến trúc CNN:
# Conv2D -> BatchNormalization -> MaxPooling -> Dropout (lặp 3 lần)
# -> Flatten -> Dense(256) -> Dropout -> Dense(5, Softmax)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# optimizer='adam': thuật toán tối ưu Adam.
# loss='categorical_crossentropy': hàm mất mát cho bài toán phân loại nhiều lớp.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==================== CALLBACKS ====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_emotion_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ==================== HUẤN LUYỆN ====================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# ==================== LƯU MÔ HÌNH CUỐI CÙNG ====================
model.save('emotion_model_5classes.keras')
print("Đã lưu mô hình vào emotion_model_5classes.keras")

with open('emotion_labels.json', 'w', encoding='utf-8') as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=2)
print("Đã lưu thứ tự nhãn vào emotion_labels.json")

# ==================== VẼ BIỂU ĐỒ ====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()
