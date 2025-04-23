import torch
import matplotlib.pyplot as plt
import numpy as np

# === Load the .pt file safely ===
pt_path = 'rosbag_data_2.pt'

try:
    data = torch.load(pt_path, map_location='cpu')  # Avoid GPU issues
except Exception as e:
    print(f"❌ Failed to load {pt_path}: {e}")
    exit()

print(f"✅ Loaded '{pt_path}'")

# === Print what's inside ===
print("📦 Keys in file:", list(data.keys()))

images = data.get('images', [])
poses = data.get('poses', [])

print(f"🖼️  Number of images: {len(images)}")
print(f"📍 Number of poses:  {len(poses)}")

if images and isinstance(images[0], (list, tuple)):
    print(f"🔍 Example image shape: {images[0][1].shape} (should be C x H x W)")
    print(f"🕒 Timestamp: {images[0][0]}")
else:
    print("⚠️ Image format may be unexpected.")

# === Visualize first image ===
if images:
    img_tensor = images[0][1]  # Should be C x H x W
    img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

    plt.imshow(img_np)
    plt.title(f"Timestamp: {images[0][0]}")
    plt.axis("off")
    plt.savefig("mushr_pictures.png")
    print("✅ Image saved as mushr_pictures.png")
else:
    print("⚠️ No images to visualize.")

print(f"✅ {len(images)} images, {len(poses)} poses saved")
print(f"🧠 Sample image shape: {images[0][1].shape if images else 'N/A'}")
