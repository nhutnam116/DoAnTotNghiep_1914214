from flask import Flask, Response
from picamera2 import Picamera2, Preview
import cv2
import io
import time

app = Flask(__name__)
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (1296, 972)})
picam2.configure(video_config)
picam2.start()

def gen_frames():
    while True:
        # Lấy khung hình từ camera
        frame = picam2.capture_array()
        # Chuyển đổi không gian màu từ BGR sang RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Mã hóa khung hình sang định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        # Trả dữ liệu khung hình dưới dạng luồng
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
