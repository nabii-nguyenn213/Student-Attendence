# import os
# import shutil
# import cv2
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
# from sklearn.model_selection import train_test_split

# # Paths
# dataset_path = "dataset/students"
# train_path = os.path.join(dataset_path, "train")
# val_path = os.path.join(dataset_path, "val")
# test_path = os.path.join(dataset_path, "test")

# # Ensure output folders exist
# for path in [train_path, val_path, test_path]:
#     os.makedirs(path, exist_ok=True)

# # Define augmentation techniques using Albumentations
# augmenter = A.Compose([
#     A.HorizontalFlip(p=0.5),  # Lật ngang
#     A.Rotate(limit=15, p=0.5),  # Xoay góc ±15 độ
#     A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Làm mờ nhẹ
#     A.RandomBrightnessContrast(p=0.5),  # Điều chỉnh độ sáng, tương phản
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Thêm nhiễu Gaussian
# ])

# # Function to augment and split data
# def augment_and_split(input_folder, num_augmented=20):
#     for student_id in os.listdir(input_folder):  # Loop qua từng student
#         student_path = os.path.join(input_folder, student_id)
#         if not os.path.isdir(student_path):
#             continue  # Bỏ qua nếu không phải folder

#         images = [f for f in os.listdir(student_path) if f.endswith(('.jpg', '.png'))]
#         if len(images) == 0:
#             print("There is no image!")
#             continue  # Bỏ qua nếu không có ảnh
        
#         img_path = os.path.join(student_path, images[0])  # Lấy ảnh gốc
#         img = cv2.imread(img_path)  # Đọc ảnh
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB

#         augmented_images = []
#         for i in range(num_augmented):
#             augmented = augmenter(image=img)["image"]  # Apply augmentation
#             augmented_images.append(Image.fromarray(augmented))  # Convert to PIL
        
#         # Chia thành train (16), val (2), test (2)
#         train_imgs, temp_imgs = train_test_split(augmented_images, test_size=0.2, random_state=42)
#         val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

#         # Save images
#         for i, img in enumerate(train_imgs):
#             save_path = os.path.join(train_path, student_id)
#             os.makedirs(save_path, exist_ok=True)
#             img.save(os.path.join(save_path, f"aug_{i}.jpg"))

#         for i, img in enumerate(val_imgs):
#             save_path = os.path.join(val_path, student_id)
#             os.makedirs(save_path, exist_ok=True)
#             img.save(os.path.join(save_path, f"aug_{i}.jpg"))

#         for i, img in enumerate(test_imgs):
#             save_path = os.path.join(test_path, student_id)
#             os.makedirs(save_path, exist_ok=True)
#             img.save(os.path.join(save_path, f"aug_{i}.jpg"))

#         print(f"Processed {student_id}: Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

# # Run augmentation and splitting
# augment_and_split(dataset_path)

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

