import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo các module của MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Mở video hoặc webcam
cap = cv2.VideoCapture(0)  # Sử dụng 0 để lấy webcam mặc định

# Khởi tạo đối tượng Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi ảnh từ BGR sang RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Xử lý ảnh để nhận diện khung sương
        results = pose.process(image)

        # Vẽ khung sương lên ảnh gốc
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Lấy vị trí các điểm landmark của mắt
            left_eye = results.pose_landmarks.landmark[2]
            right_eye = results.pose_landmarks.landmark[5]

            # Vẽ điểm landmark của mắt
            eye_radius = 5
            cv2.circle(image, (int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])), eye_radius, (0, 255, 0), -1)
            cv2.circle(image, (int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])), eye_radius, (0, 255, 0), -1)

            # Ước tính hướng nhìn của mắt
            eye_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
            eye_direction = eye_vector / np.linalg.norm(eye_vector)

            # Tính góc giữa vector hướng nhìn và trục ngang
            angle = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))

            # Hiển thị hướng nhìn của mắt
            gaze_length = 50  # Độ dài của vector hướng nhìn
            left_eye_center = (int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0]))
            right_eye_center = (int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0]))

            left_gaze_point = (int(left_eye_center[0] + gaze_length * eye_direction[0]),
                               int(left_eye_center[1] + gaze_length * eye_direction[1]))
            right_gaze_point = (int(right_eye_center[0] + gaze_length * eye_direction[0]),
                                int(right_eye_center[1] + gaze_length * eye_direction[1]))

            cv2.line(image, left_eye_center, left_gaze_point, (255, 0, 0), 2)
            cv2.line(image, right_eye_center, right_gaze_point, (255, 0, 0), 2)

            # In góc nhìn lên ảnh
            angle_text = f'Angle: {angle:.2f} degrees'
            cv2.putText(image, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Hiển thị ảnh kết quả
        cv2.imshow('MediaPipe Pose with Eye Gaze', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
