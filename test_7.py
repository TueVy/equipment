import cv2
import mediapipe as mp
import numpy as np

# Hàm tính góc giữa hai vector
def calculate_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product) * (180 / np.pi)
    return angle

# Khởi tạo MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo video
video_path = 'videoplayback.mp4'  # Thay 'path_to_your_video.mp4' bằng đường dẫn tới tệp video của bạn
output_path = 'output_video.mp4'  # Đường dẫn để lưu video đã xử lý
cap = cv2.VideoCapture(video_path)

# Lấy thông tin về video để thiết lập đầu ra
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Khởi tạo VideoWriter để lưu video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video (mp4v cho mp4)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Chuyển đổi màu từ BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Nhận diện khung xương
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Vẽ khung xương
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
        
        # Lấy vị trí các điểm mốc quan trọng
        left_eye = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE]
        right_eye = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE]
        nose_tip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        
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
        cv2.putText(image, f"Angle: {angle:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Cảnh báo nếu góc ngoài phạm vi 45 - 135 độ
        if angle < 45 or angle > 135:
            cv2.putText(image, "ALERT: Head Turned", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Ghi khung hình vào video đầu ra
    out.write(image)
    
    # Hiển thị hình ảnh
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Đóng tất cả
cap.release()
out.release()
cv2.destroyAllWindows()
