import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "dataset/students"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

# Ensure output folders exist
for path in [train_path, val_path, test_path]:
    os.makedirs(path, exist_ok=True)

# Define augmentation techniques using Albumentations
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
])

def read_image_unicode(path):
    """ Đọc ảnh có tên file Unicode bằng OpenCV """
    try:
        with open(path, "rb") as f:
            img_data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Lỗi đọc ảnh")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Lỗi đọc ảnh: {path} ({e})")
        return None

def augment_and_split(input_folder, num_augmented=20):
    for entry in os.scandir(input_folder):  # Duyệt thư mục với os.scandir() hỗ trợ Unicode
        if not entry.is_dir() or entry.name in ["train", "val", "test"]:
            continue  # Bỏ qua nếu không phải thư mục sinh viên

        student_id = entry.name
        student_path = entry.path  # Unicode path

        images = [f for f in os.listdir(student_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            print(f"⚠ Không có ảnh trong thư mục {student_id}, bỏ qua...")
            continue

        img_path = os.path.join(student_path, images[0])
        img = read_image_unicode(img_path)
        if img is None:
            continue  # Bỏ qua nếu ảnh lỗi

        # Generate augmented images
        augmented_images = [Image.fromarray(augmenter(image=img)["image"]) for _ in range(num_augmented)]

        # Split into train (16), val (2), test (2)
        train_imgs, temp_imgs = train_test_split(augmented_images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        # Save images
        for img_list, folder in zip([train_imgs, val_imgs, test_imgs], [train_path, val_path, test_path]):
            save_path = os.path.join(folder, student_id)
            os.makedirs(save_path, exist_ok=True)
            for i, img in enumerate(img_list):
                img.save(os.path.join(save_path, f"aug_{i}.jpg"))

        print(f"✅ {student_id}: Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

# Run
augment_and_split(dataset_path)

