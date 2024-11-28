import matplotlib.pyplot as plt

# Số lượng hình ảnh trong từng tập
train_set = 2520
valid_set = 224
test_set = 173

# Tính toán tổng số hình ảnh
total_images = train_set + valid_set + test_set

# Các tỷ lệ tương ứng với từng tập
labels = ['Train Set', 'Valid Set', 'Test Set']
sizes = [train_set, valid_set, test_set]
colors = ['#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0)  # Tạo hiệu ứng cho phần "Train Set" nhô ra

# Vẽ biểu đồ hình tròn
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# Thêm tiêu đề
plt.title(f"Dataset Distribution: {total_images} Images")

# Hiển thị biểu đồ
plt.axis('equal')  # Đảm bảo hình tròn cân đối
plt.show()
