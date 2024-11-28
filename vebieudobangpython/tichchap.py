import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Khởi tạo dữ liệu cho hình ảnh đầu vào (input), kernel và output
input_matrix = np.array([
    [1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0]
])

kernel = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

output_matrix = np.zeros((4, 4), dtype=int)

# Tính toán tích chập cho ô đầu tiên (giá trị đầu tiên của output)
image_patch = input_matrix[0:3, 0:3]
convolved_value = np.sum(image_patch * kernel)
output_matrix[0, 0] = convolved_value

# Thiết lập biểu đồ
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Hiển thị input matrix với vùng được chọn
ax[0].imshow(input_matrix, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Input")
for (i, j), val in np.ndenumerate(input_matrix):
    ax[0].text(j, i, f"{val}", ha='center', va='center', color="black")
# Vẽ hình chữ nhật xanh dương quanh vùng chọn
rect = Rectangle((0.5, 0.5), 3, 3, linewidth=2, edgecolor='blue', facecolor='none')
ax[0].add_patch(rect)

# Hiển thị image patch và kernel
ax[1].imshow(np.zeros((3, 6)), cmap="gray")  # Nền màu xám
ax[1].set_title("Image Patch & Kernel")
for (i, j), val in np.ndenumerate(image_patch):
    ax[1].text(j, i, f"{val}", ha='center', va='center', color="black")
for (i, j), val in np.ndenumerate(kernel):
    ax[1].text(j + 3, i, f"{val}", ha='center', va='center', color="black")

# Vẽ dấu nhân "x" giữa image patch và kernel
ax[1].text(2.5, 1, "×", fontsize=15, ha='center', va='center', color="blue")

# Hiển thị output matrix với giá trị đầu tiên đã được tính
ax[2].imshow(output_matrix, cmap="gray", vmin=0, vmax=50)
ax[2].set_title("Output")
for (i, j), val in np.ndenumerate(output_matrix):
    ax[2].text(j, i, f"{val if val != 0 else ''}", ha='center', va='center', color="black")
# Vẽ hình chữ nhật xanh dương quanh ô đầu tiên của output
rect_output = Rectangle((0.5, 0.5), 1, 1, linewidth=2, edgecolor='blue', facecolor='none')
ax[2].add_patch(rect_output)

# Ẩn trục
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
