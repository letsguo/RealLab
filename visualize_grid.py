import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# === Load data ===
data = torch.load('rosbag_data_2.pt', map_location='cpu')
images = data['images']  # list of (timestamp, image_tensor)

# === Config ===
num_images = len(images)
cols = 10
rows = math.ceil(num_images / cols)
figsize = (cols * 2, rows * 2)  # tweak as needed

# === Plot ===
fig, axs = plt.subplots(rows, cols, figsize=figsize)
axs = axs.flatten()

for i, (ts, img_tensor) in enumerate(images):
    img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    axs[i].imshow(img_np)
    axs[i].set_title(f"{ts:.2f}", fontsize=8)
    axs[i].axis('off')

# Hide extra axes (if total < rows*cols)
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("rosbag_image_grid.png", dpi=300)
print("✅ Saved full image grid to: rosbag_image_grid.png")
