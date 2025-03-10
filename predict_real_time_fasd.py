import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã train
model = load_model("models/CNN.keras")

# Khởi động webcam
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Resize về đúng kích thước mô hình yêu cầu
    img = cv2.resize(frame, (64, 64))  # Kích thước ảnh khi train model
    img = img.astype("float32") / 255.0  # Chuẩn hóa về [0, 1]
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Dự đoán bằng mô hình 
    prediction = model.predict(img)[0]
    print(prediction)
    label = "Spoof" if prediction > 0.5 else "Real"
    # Hiển thị kết quả lên màn hình
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Face Anti-Spoofing", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
