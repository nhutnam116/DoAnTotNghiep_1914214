import ultralytics
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, render_template, Response
import random
import torch
# Khởi tạo Flask app
app = Flask(__name__)

# Khởi tạo mô hình YOLO
model = YOLO(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11s\train\weights\best.pt")

# Lấy danh sách tên các lớp từ mô hình
class_names = model.names

# Khởi tạo bộ theo dõi Deep SORT
tracker = DeepSort(max_iou_distance=0.7, max_age=10, n_init=3, nms_max_overlap=0.5)

# Tạo một từ điển để ánh xạ class_id tới màu sắc
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

class_colors = {}
for cls_id in range(len(class_names)):
    class_colors[cls_id] = random_color()

# Đường dẫn tới video
video_path = 'http://192.168.137.34:8000/video_feed'
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Kiểm tra xem CUDA có sẵn không
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Khởi tạo VideoWriter để ghi video kết quả
output_path = 'output_with_tracking.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Hàm để lấy frame từ video và gửi cho Flask
def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện đối tượng bằng YOLO
        results = model.predict(source=frame, conf=0.25)

        # Lấy thông tin các đối tượng phát hiện được
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                left = x1.item()
                top = y1.item()
                width_box = (x2 - x1).item()
                height_box = (y2 - y1).item()
                detections.append([[left, top, width_box, height_box], confidence.item(), class_id])

        # Cập nhật bộ theo dõi
        tracks = tracker.update_tracks(detections, frame=frame)

        # Vẽ kết quả theo dõi lên khung hình
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.det_class
            class_name = class_names[class_id] if class_id in class_names else 'Unknown'
            color = class_colors[class_id] if class_id in class_colors else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            label = f'ID: {track_id} {class_name}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Chuyển frame thành định dạng JPEG để phát trên Flask
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route chính để hiển thị video
@app.route('/')
def index():
    return render_template('index.html')  # Tạo file index.html để hiển thị video

# Route để phát video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Giải phóng tài nguyên
cap.release()
out.release()
