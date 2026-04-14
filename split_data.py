import os
import shutil
import random

# Đường dẫn thư mục gốc (chứa Angry, Fear, ...)
source_dir = 'Data'   # Nếu thư mục tên là 'Data' thì giữ nguyên, nếu khác thì sửa

# Thư mục đích sẽ tạo
dest_dir = 'face_emotion_split'

# Tỉ lệ train 80%, test 20%
train_ratio = 0.8

# Tạo cấu trúc thư mục đích
os.makedirs(dest_dir, exist_ok=True)
for split in ['train', 'test']:
    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

# Lấy danh sách các cảm xúc (thư mục con trong source_dir)
emotions = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for emo in emotions:
    # Tạo thư mục cảm xúc trong train và test
    os.makedirs(os.path.join(dest_dir, 'train', emo), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test', emo), exist_ok=True)
    
    # Lấy tất cả file ảnh (jpg, png, jpeg) trong thư mục cảm xúc
    img_files = [f for f in os.listdir(os.path.join(source_dir, emo))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Xáo trộn ngẫu nhiên
    random.shuffle(img_files)
    
    # Tính số lượng cho train
    train_count = int(len(img_files) * train_ratio)
    train_files = img_files[:train_count]
    test_files = img_files[train_count:]
    
    # Copy file
    for f in train_files:
        shutil.copy(os.path.join(source_dir, emo, f), os.path.join(dest_dir, 'train', emo, f))
    for f in test_files:
        shutil.copy(os.path.join(source_dir, emo, f), os.path.join(dest_dir, 'test', emo, f))
    
    print(f"{emo}: {len(train_files)} train, {len(test_files)} test")

print("Hoàn tất! Dữ liệu đã được chia vào thư mục 'face_emotion_split'")