import cv2
import numpy as np
import joblib
from ultralytics import YOLO

class PhoneSpoofDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.phone_conf = None

    def process_frame(self, frame):
        results = self.model(frame, conf=0.2, verbose=False)

        # Extract phone detections with confidence scores
        phone_detections = [
            (box.conf[0].item(), box.xyxy[0].tolist())  # (confidence, bounding box)
            for r in results for box in r.boxes if int(box.cls) == 67
        ]

        phone_detected = bool(phone_detections)
        self.phone_conf = max([conf for conf, _ in phone_detections], default=0.0)  # Highest confidence

        return phone_detected

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            detections = self.process_frame(frame)

            print(f'phone confidence: {self.phone_conf:.2f}')
            if detections:
                text = f"PHONE SPOOF DETECTED! ({self.phone_conf:.2%})"
                color = (0, 0, 255)
            else:
                text = "NO SPOOF DETECTED"
                color = (0, 255, 0)

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Phone Spoof Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_model(self, filename="phone_spoof_detector.pkl"):
        joblib.dump(self, filename)
        print(f"Successfully save model to {filename}")

    @staticmethod
    def load_model(filename="phone_spoof_detector.pkl"):
        return joblib.load(filename)

if __name__ == "__main__":
    detector = PhoneSpoofDetector()
    detector.run()
    detector.save_model(filename='models/phone_detector.pkl')
