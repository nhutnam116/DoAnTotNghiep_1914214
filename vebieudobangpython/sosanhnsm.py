import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ các file CSV (thay đổi đường dẫn file nếu cần)
df_n = pd.read_csv(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11m\train\results.csv")
df_s = pd.read_csv(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11s\train\results.csv")
df_m = pd.read_csv(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11n\train\results.csv")

# Lấy số lượng epoch (dòng dữ liệu) - giả sử mỗi file CSV có 100 dòng (1 dòng mỗi epoch)
epochs = df_n['epoch']  # Sử dụng 'epoch' từ file N, tất cả các file đều có cùng số epoch

# Lấy thời gian huấn luyện cho mỗi epoch từ các variant
time_n = df_n['time']
time_s = df_s['time']
time_m = df_m['time']

# Vẽ biểu đồ so sánh thời gian huấn luyện qua từng epoch
plt.figure(figsize=(10, 6))

# Vẽ các đường cho từng variant
plt.plot(epochs, time_n, label='Variant m', color='blue', marker='o', linestyle='-')
plt.plot(epochs, time_s, label='Variant s', color='green', marker='x', linestyle='--')
plt.plot(epochs, time_m, label='Variant n', color='orange', marker='^', linestyle='-.')

# Thêm tiêu đề và nhãn
plt.title('So sánh Thời gian Huấn luyện YOLO11 qua Từng Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Thời gian (Giây)', fontsize=12)
plt.legend(title="Variants", loc='upper right')

# Hiển thị biểu đồ
plt.grid(True)
plt.show()
