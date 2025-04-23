#!/usr/bin/env python3
import rospy
import rosbag
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

class RosbagRecorder:
    def __init__(self, bag_path):
        self.bag = rosbag.Bag(bag_path, 'w')
        # self.odom_topic = odom_topic

        # active flag
        self.active = True

        # Subscribe to topics
        self.img_sub  = rospy.Subscriber('/camera/color/image_raw', Image,   self.image_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber('/car/odom',               Odometry, self.odom_cb, queue_size=1)
        self.imu_sub  = rospy.Subscriber('/camera/gyro/sample',      Imu,     self.imu_cb, queue_size=1)

        # Ensure bag is closed on shutdown
        rospy.on_shutdown(self.shutdown_cb)
        # rospy.spin()

    def image_cb(self, msg):
        # Directly write the raw Image message
        if not self.active:
            return
        self.bag.write('/camera/color/image_raw', msg, msg.header.stamp)

    def odom_cb(self, msg):
        # Write Odometry with its header timestamp
        if not self.active:
            return
        self.bag.write('/car/odom', msg, msg.header.stamp)

    def imu_cb(self, msg):
        # Write IMU
        if not self.active:
            return
        self.bag.write('/camera/gyro/sample', msg, msg.header.stamp)


    def shutdown_cb(self):
        # stop new callbacks
        self.active = False

        # unregister so ROS won’t queue any more
        self.image_sub.unregister()
        self.odom_sub.unregister()
        self.imu_sub.unregister()

        # give any in‑flight callbacks a moment
        rospy.sleep(0.1)

        # close the bag safely
        self.bag.close()
        rospy.loginfo("RosbagRecorder: bag closed cleanly.")


if __name__ == '__main__':
    # bag_file   = sys.argv[1]
    # odom_topic = sys.argv[2]
    rospy.init_node('img_recorder_rosbag')
    bag_file   = rospy.get_param('~bag_file',   '/tmp/run.bag')

    recorder = RosbagRecorder(bag_file)
    rospy.spin()
