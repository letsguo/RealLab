import rosbag
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2

# === CONFIG ===
bag_path = 'data_2.bag'             # Replace with your .bag file
output_pt = 'rosbag_data_3.pt'     # Output .pt file
image_topic = '/camera/color/image_raw'
odom_topic = '/car/odom'

# === SETUP ===
bridge = CvBridge()
images = []
poses = []

# === READ BAG ===
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[image_topic, odom_topic]):
        timestamp = t.to_sec()

        if topic == image_topic:
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # shape: H x W x 3
                img_tensor = torch.from_numpy(cv_image).permute(2, 0, 1)       # C x H x W
                images.append((timestamp, img_tensor))
            except Exception as e:
                print(f"[Image Error] {e}")

        elif topic == odom_topic:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            velocity = msg.twist.twist.linear
            pose_tensor = torch.tensor([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w, velocity.x, velocity.y, velocity.z])
            poses.append((timestamp, pose_tensor))

# === SAVE TO .pt ===
torch.save({'images': images, 'poses': poses}, output_pt)
print(f"✅ Saved data to {output_pt}")
