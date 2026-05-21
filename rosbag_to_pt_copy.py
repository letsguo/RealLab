import rosbag
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2
import os

# === CONFIG ===
bag_folder= 'gail_demonstrations'
output_pt_folder = 'gail_pt'
odom_topic = '/mocap/local_position/odom'

# === SETUP ===
bridge = CvBridge()
states = []

odom_msgs = []

# === PROCESS ALL BAG FILES ===
for filename in os.listdir(bag_folder):
    if filename.endswith('.bag'):
        bag_path = os.path.join(bag_folder, filename)
        base_name = os.path.splitext(filename)[0]  # '4_24_4pm' from '4_24_4pm.bag'
        output_pt = os.path.join(output_pt_folder, f'{base_name}.pt')

        print(f"Processing {bag_path} -> {output_pt}")

        states = []

        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[odom_topic]):
                timestamp = t.to_sec()

                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                lin_vel = msg.twist.twist.linear
                # ang_vel = msg.twist.twist.angular  # <-- corrected (was msg.angular_velocity)

                state_tensor = torch.tensor([
                    pos.x, pos.y, pos.z,
                    ori.x, ori.y, ori.z, ori.w,
                    lin_vel.x, lin_vel.y, lin_vel.z,
                ])
                # state_tensor = torch.tensor([
                #     pos.x, pos.y, pos.z,
                #     lin_vel.x, lin_vel.y, lin_vel.z,
                # ])
                states.append((timestamp, state_tensor))

        torch.save({'states': states}, output_pt)

print("All bag files processed!")