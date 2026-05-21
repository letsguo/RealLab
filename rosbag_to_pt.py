import rosbag
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2

# === CONFIG ===
bag_path = '4_24_4pm.bag'
output_pt = 'rosbag_data_4_24_4pm.pt'
image_topic = '/camera/color/image_raw'
odom_topic = '/car/odom'
imu_topic = '/camera/gyro/sample'

# === SETUP ===
bridge = CvBridge()
images = []
states = []

odom_msgs = []
imu_msgs = []

# === READ BAG ===
with rosbag.Bag(bag_path, 'r') as bag:

    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        odom_msgs.append((t.to_sec(), msg))

    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        imu_msgs.append((t.to_sec(), msg))

    odom_idx = 0
    imu_idx = 0

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        timestamp = t.to_sec()

        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # shape: H x W x 3
            img_tensor = torch.from_numpy(cv_image).permute(2, 0, 1)       # C x H x W
            images.append((timestamp, img_tensor))
        except Exception as e:
            print(f"[Image Error] {e}")

        odom_time, odom_msg = odom_msgs[odom_idx]
        imu_time, imu_msg = imu_msgs[imu_idx]

        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        lin_vel = odom_msg.twist.twist.linear
        ang_vel = imu_msg.angular_velocity

        state_tensor = torch.tensor([
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w,
            lin_vel.x, lin_vel.y, lin_vel.z,
            ang_vel.x, ang_vel.y, ang_vel.z
        ])
        states.append((timestamp, state_tensor))

        odom_idx += 1
        imu_idx += 1

# === SAVE TO .pt ===
torch.save({'images': images, 'states': states}, output_pt)
print(f"Saved data to {output_pt}")
