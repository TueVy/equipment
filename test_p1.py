import cv2
import mediapipe as mp
import numpy as np
import signal
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
# Hàm tính góc giữa hai vector
def calculate_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product) * (180 / np.pi)
    return angle

# Hàm xử lý tín hiệu dừng
def signal_handler(sig, frame):
    print("Program received signal to stop")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm chạy xử lý video và hiển thị trong giao diện Tkinter
def update_frame():
    success, image = cap.read()
    if success:
        # Chuyển đổi màu từ BGR sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Nhận diện khung xương
        results = pose.process(image)
        image.flags.writeable = True
        
        if results.pose_landmarks:
            # Vẽ khung xương
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            
            # Lấy vị trí các điểm mốc quan trọng
            left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
            nose_tip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            
            # Tính vector hướng nhìn
            eye_center = np.array([(left_eye.x + right_eye.x) * 0.5, (left_eye.y + right_eye.y) * 0.5])
            nose_vector = np.array([nose_tip.x, nose_tip.y]) - eye_center
            horizontal_vector = np.array([1, 0])
            
            # Tính góc hướng nhìn
            angle = calculate_angle(nose_vector, horizontal_vector)
            
            # Vẽ hướng nhìn và cảnh báo
            eye_center_pixel = (int(eye_center[0] * image.shape[1]), int(eye_center[1] * image.shape[0]))
            nose_tip_pixel = (int(nose_tip.x * image.shape[1]), int(nose_tip.y * image.shape[0]))
            cv2.arrowedLine(image, eye_center_pixel, nose_tip_pixel, (0, 255, 0), 2)
            
            # Hiển thị góc hướng nhìn
            cv2.putText(image, f"Angle: {angle:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            # Cảnh báo nếu góc ngoài phạm vi 45 - 135 độ
            if angle < 65 or angle > 105:
                cv2.putText(image, "Attitude warning", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Chuyển đổi hình ảnh thành định dạng để hiển thị trong Tkinter
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    root.after(10, update_frame)

# Hàm mở giao diện video
def open_video():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

# Hàm đóng ứng dụng
def close_app():
    if messagebox.askokcancel("Quit", "Bạn muốn đóng?"):
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()
def open_new_program():
    subprocess.Popen(['python', 'test_p2.py'])  # Thay 'path_to_your_script.py' bằng đường dẫn thực tế
    root.destroy()

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Giám sát thi")

# open_button = tk.Button(root, text="Open Video", command=open_video)
# open_button.pack(side=tk.LEFT, padx=10, pady=10)

close_button = tk.Button(root, text="Đóng", command=close_app)
close_button.pack(side=tk.LEFT, padx=10, pady=10)
open_button = tk.Button(root, text="Mở nhận diện nề nếp", command=open_new_program)
open_button.pack(side=tk.LEFT, padx=10, pady=10)
lbl_video = tk.Label(root)
lbl_video.pack()

root.protocol("WM_DELETE_WINDOW", close_app)
open_video()
root.mainloop()
