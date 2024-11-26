import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo các đối tượng MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Đọc hình ảnh
image_path = 'test.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Xử lý hình ảnh để nhận diện khớp xương
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

# Vẽ khớp xương lên ảnh và tính toán góc nhìn
if results.pose_landmarks:
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Lấy tọa độ của vai trái, vai phải và đầu
    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Chuyển đổi tọa độ thành pixel
    h, w, _ = annotated_image.shape
    left_shoulder_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
    right_shoulder_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
    nose_pixel = (int(nose.x * w), int(nose.y * h))

    # Tính toán góc nhìn dựa trên vị trí của vai và đầu
    angle = np.arctan2(right_shoulder_pixel[1] - left_shoulder_pixel[1], right_shoulder_pixel[0] - left_shoulder_pixel[0])
    angle_degrees = np.degrees(angle)

    # Hiển thị góc nhìn lên ảnh
    cv2.putText(annotated_image, f"Angle: {angle_degrees:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Lưu ảnh đã được chú thích
    output_path = 'annotated_image_with_angle.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"Image saved to {output_path}")
else:
    print("No pose landmarks detected.")
