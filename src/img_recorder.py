#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, Image, Joy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from utils.rl_policy import RLModel
from utils.waypoints import Waypoints
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion

import os
from pathlib import Path
import yaml
import time
import torch
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from utils.generate_elevation_map import crop_heightmap


class Image_Recorder:
    def __init__(self, odom_topic):
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
            
        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback
        )

        self.imu_sub = rospy.Subscriber("/camera/gyro/sample", Imu, self.imu_callback)
        self.data = []
        self.imu = None
        self.image = np.zeros(1)
        self.odom = np.zeros(6)
        self.image_init = False
        self.odom_init = False

        # put function to run on shutdown
        rospy.on_shutdown(self.shutdown_callback)
        self.main_loop()


    
    def main_loop(self):
        rate = rospy.Rate(10) # same rate as sim
        while not rospy.is_shutdown():
            # write image and odom to self.data
            # temp = np.zeros(1)
            # temp.append(self.image)
            # temp.append(self.odom)
            # data.append(temp)
            if self.image_init and self.odom_init:
                self.data.append(np.concatenate((self.odom, self.image), axis=0))

            rate.sleep()

    def image_callback(self, msg):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # print(image.ty)
            image = np.array(image)
            self.image = image.flatten() # flatten image
            self.image_init = True

        except CvBridgeError as e:
            raise RuntimeError(e)


        
        # h x w x c
        

    def odom_callback(self, odom):
        if self.imu is None:
            return
        self.odom[0] = odom.twist.twist.linear.x
        self.odom[1] = odom.twist.twist.linear.y
        self.odom[2] = odom.twist.twist.linear.z
        # lazy fix for wierd camera reference frame
        self.odom[3] = self.imu.angular_velocity.z
        self.odom[4] = - self.imu.angular_velocity.x
        self.odom[5] = - self.imu.angular_velocity.y

        self.odom_init = True

    def imu_callback(self, imu):
        self.imu = imu

    def shutdown_callback(self):
        data = np.stack(self.data)
        print('a')
        np.save("/root/catkin_ws/src/data.npy", data)
        # self.data.save("/root/catkin_ws/data/data.npy")


if __name__ == "__main__":
    rospy.init_node("img_recorder")
    planner = Image_Recorder('/car/vesc/odom')
    rospy.spin()
