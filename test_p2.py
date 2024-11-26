import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import subprocess
from tkinter import messagebox
# Load mô hình YOLOv5 tùy chỉnh
model_path = 'best_door.pt'  # Thay đổi đường dẫn đến file model của bạn
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = None
running = False

def start_detection():
    global running, cap, label
    running = True
    cap = cv2.VideoCapture(0)  # '0' là ID của camera mặc định
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set chiều rộng
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set chiều cao

    while running:
        # Đọc một khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Chuyển đổi frame sang định dạng mong muốn của YOLOv5 và nhận diện
        results = model(frame)

        # Chuyển kết quả nhận diện thành numpy array
        img = np.squeeze(results.render())

        # Chuyển đổi khung hình từ OpenCV sang định dạng của Tkinter
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Hiển thị hình ảnh trên nhãn Tkinter
        label.imgtk = img
        label.configure(image=img)

    cap.release()

def stop_detection():
    global running
    running = False
    if cap is not None:
        cap.release()
def close_app():
    if messagebox.askokcancel("Quit", "Bạn muốn đóng?"):
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()
def start_thread():
    detection_thread = Thread(target=start_detection)
    detection_thread.start()

def open_new_program():
    subprocess.Popen(['python', 'test_p1.py'])  # Thay 'path_to_your_script.py' bằng đường dẫn thực tế
    root.destroy()

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Nhận diện nề nếp")

# Tạo nhãn để hiển thị video
label = tk.Label(root)
label.pack()

# Tạo các nút
# start_button = tk.Button(root, text="Mở", command=start_thread)
# start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = tk.Button(root, text="Đóng", command=close_app)
stop_button.pack(side=tk.LEFT, padx=10, pady=10)

# Nút để mở chương trình Python khác
open_button = tk.Button(root, text="Mở Giám Sát Thi", command=open_new_program)
open_button.pack(side=tk.LEFT, padx=10, pady=10)
start_thread()
# Chạy ứng dụng
root.mainloop()
