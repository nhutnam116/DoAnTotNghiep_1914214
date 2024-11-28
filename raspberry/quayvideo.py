from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from datetime import datetime
import time

def main():
    # Khởi tạo Picamera2
    picam2 = Picamera2()

    # Cấu hình preview
    preview_config = picam2.create_preview_configuration(main={"size": (1296, 972)})
    picam2.configure(preview_config)

    # Bật preview (tuỳ chọn, có thể bỏ)
    #picam2.start_preview(Preview.QTGL)

    # Tạo encoder với bitrate
    encoder = H264Encoder(5000000)  # Bitrate 5 Mbps

    # Bắt đầu camera
    picam2.start()

    # Tạo tên file video với timestamp
    filename = f"/home/admin/Desktop/video_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

    # Sử dụng FfmpegOutput để đóng gói tệp h264 trực tiếp thành MP4
    video_output = FfmpegOutput(filename, audio=False)

    print(f"Starting video recording, saving to {filename}")

    # Bắt đầu ghi video
    picam2.start_recording(encoder, output=video_output)

    # Thời gian quay (10 phút = 600 giây)
    time.sleep(600)

    # Dừng ghi video
    picam2.stop_recording()
    print(f"Recording finished. Video saved to {filename}")

    # Tắt camera
    picam2.stop()

if __name__ == '__main__':
    main()

