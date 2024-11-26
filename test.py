import cv2
import torch
import numpy as np

# Load mô hình YOLOv5 tùy chỉnh
model_path = 'best_door.pt'  # Thay đổi đường dẫn đến file model của bạn
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Khởi tạo camera
cap = cv2.VideoCapture(0)  # '0' là ID của camera mặc địnhq
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set chiều rộng
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set chiều cao

try:
    while True:
        # Đọc một khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Chuyển đổi frame sang định dạng mong muốn của YOLOv5 và nhận diện
        results = model(frame)

        # Chuyển kết quả nhận diện thành numpy array và hiển thị
        cv2.imshow('YOLOv5 Custom Model Detection', np.squeeze(results.render()))

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Giải phóng và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
