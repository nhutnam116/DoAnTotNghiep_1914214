import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load ảnh mẫu dưới dạng màu
image = cv2.imread(r"C:\Users\Admin\Downloads\catest2.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB cho hiển thị màu sắc

# Định nghĩa các bộ lọc và tên tương ứng
filters = {
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "Edge Detection 1": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "Edge Detection 2": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Edge Detection 3": np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Box Blur": np.ones((3, 3)) / 9,  # Lọc trung bình (mờ hộp)
    "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # Lọc Gaussian xấp xỉ
}

# Hàm để áp dụng bộ lọc
def apply_filter(image, kernel, keep_color=False):
    if keep_color:
        # Áp dụng bộ lọc trên từng kênh màu R, G, B riêng biệt
        filtered_channels = [cv2.filter2D(image[:, :, i], ddepth=-1, kernel=kernel) for i in range(3)]
        filtered_image = cv2.merge(filtered_channels)
    else:
        # Chuyển ảnh sang grayscale và áp dụng bộ lọc
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filtered_image = cv2.filter2D(src=gray_image, ddepth=-1, kernel=kernel)
    return filtered_image

# Hiển thị kết quả theo dạng bảng
plt.figure(figsize=(12, 18))  # Giảm chiều rộng và chiều cao tổng thể

for i, (name, kernel) in enumerate(filters.items()):
    # Chọn bộ lọc có giữ lại màu sắc hay không
    keep_color = name not in ["Edge Detection 1", "Edge Detection 2", "Edge Detection 3"]
    filtered_image = apply_filter(image_rgb, kernel, keep_color=keep_color)
    
    # Hiển thị tên bộ lọc (Operation)
    plt.subplot(len(filters), 3, 3*i + 1)
    plt.text(0.5, 0.5, name, fontsize=12, ha='center', va='center')
    plt.axis('off')
    
    # Hiển thị ma trận bộ lọc (Filter) dưới dạng các con số, ẩn viền
    plt.subplot(len(filters), 3, 3*i + 2)
    plt.axis('off')
    table_text = [[f"{value:.2f}" if abs(value) < 1 else str(int(value)) for value in row] for row in kernel]
    table = plt.table(cellText=table_text, loc='center', cellLoc='center', edges='')  # Loại bỏ viền
    table.scale(0.6, 1.5)  # Giảm kích thước ô để thon gọn hơn
    for key, cell in table.get_celld().items():
        cell.set_fontsize(10)  # Giảm kích thước font chữ
        cell.set_linewidth(0)  # Ẩn viền của mỗi ô trong bảng

    # Hiển thị ảnh kết quả sau khi áp dụng bộ lọc (Convolved Image)
    plt.subplot(len(filters), 3, 3*i + 3)
    if keep_color:
        plt.imshow(filtered_image)
    else:
        plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

plt.tight_layout(pad=1.5)  # Tăng khoảng cách giữa các hàng để dễ nhìn hơn
plt.show()
