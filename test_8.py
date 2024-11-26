import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo các đối tượng của MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Khởi tạo đối tượng phát hiện tư thế và khuôn mặt
pose = mp_pose.Pose()
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh()

# Đường dẫn tới tệp video
video_path = 'test.jpg'  # Thay thế bằng đường dẫn chính xác tới tệp video của bạn
# cap = cv2.VideoCapture(video_path)
image = cv2.imread(video_path)
# Kiểm tra nếu video được mở thành công
# if not cap.isOpened():
#     raise ValueError(f"Could not open the video file: {video_path}")

def calculate_head_angle(landmarks, frame_shape):
    ih, iw, _ = frame_shape
    nose_tip = landmarks[1]  # Mũi đỉnh
    left_eye_inner = landmarks[133]  # Mắt trái trong
    right_eye_inner = landmarks[362]  # Mắt phải trong

    nose_tip_coords = (int(nose_tip.x * iw), int(nose_tip.y * ih))
    left_eye_inner_coords = (int(left_eye_inner.x * iw), int(left_eye_inner.y * ih))
    right_eye_inner_coords = (int(right_eye_inner.x * iw), int(right_eye_inner.y * ih))

    angle = np.arctan2(right_eye_inner_coords[1] - left_eye_inner_coords[1], 
                       right_eye_inner_coords[0] - left_eye_inner_coords[0])
    angle_degrees = np.degrees(angle)
    return angle_degrees

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    frame = cv2.imread(video_path)
    # Chuyển đổi khung hình từ BGR sang RGB
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    faces = face_detection.process(frame_rgb)

    # Nếu phát hiện được khuôn mặt
    if faces.detections:
        for idx, detection in enumerate(faces.detections):
            # Lấy tọa độ bounding box của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                            int(bboxC.width * iw), int(bboxC.height * ih))

            # Mở rộng vùng chọn để bao gồm cả cơ thể
            y_start = max(0, y - int(2 * h))
            y_end = min(ih, y + int(4 * h))
            x_start = max(0, x - int(w))
            x_end = min(iw, x + int(2 * w))

            # Cắt khung hình quanh vùng cơ thể
            body_img = frame[y_start:y_end, x_start:x_end]

            # Chuyển đổi khung hình từ BGR sang RGB
            body_rgb = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)

            # Phát hiện tư thế trong khung hình của cơ thể
            pose_results = pose.process(body_rgb)

            # Nếu phát hiện được tư thế
            if pose_results.pose_landmarks:
                # Vẽ khung xương lên hình ảnh
                mp_drawing.draw_landmarks(
                    body_img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Phát hiện các điểm lưới khuôn mặt
            face_mesh_results = face_mesh.process(frame_rgb)

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    angle = calculate_head_angle(face_landmarks.landmark, frame.shape)

                    # Hiển thị góc nhìn lên khung hình
                    cv2.putText(frame, f'Angle: {angle:.2f}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Hiển thị vùng chứa cơ thể của từng đối tượng
            # window_name = f'Object {idx + 1}'
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(window_name, body_img)

    # Hiển thị khung hình chính với khung xương đã vẽ và góc nhìn
    cv2.imshow('Khung xương đa đối tượng', frame)
    output_image_path = 'output_image_with_pose.jpg'
    cv2.imwrite(output_image_path, image)


    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
# cap.release()
cv2.destroyAllWindows()
