#!/usr/bin/env python3
import rospy
import rosbag
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
import threading
import message_filters

class RosbagRecorder:
    def __init__(self, bag_path):
        self.bag = rosbag.Bag(bag_path, 'w')
        # self.odom_topic = odom_topic

        self.active = True
        self.lock = threading.Lock()

        self.odom_sub = rospy.Subscriber('/car/odom', Odometry, self.synced_callback, queue_size=10)

        # Ensure bag is closed on shutdown
        rospy.on_shutdown(self.shutdown_cb)

    def synced_callback(self, odom_msg):
        if not self.active:
            return
        with self.lock:
            self.bag.write('/car/odom', odom_msg, odom_msg.header.stamp)

    def shutdown_cb(self):
        # stop new callbacks
        self.active = False

        # unregister so ROS won’t queue any more
        self.odom_sub.unregister()

        # give any in‑flight callbacks a moment
        rospy.sleep(0.1)

        # close the bag safely
        with self.lock:
            self.bag.close()
        rospy.loginfo("RosbagRecorder: bag closed cleanly.")

class MocapRecorder:
    def __init__(self, bag_path):
        self.bag = rosbag.Bag(bag_path, 'w')
        # self.odom_topic = odom_topic

        self.active = True
        self.lock = threading.Lock()

        self.odom_sub = rospy.Subscriber('/mocap/local_position/odom', Odometry, self.synced_callback, queue_size=10)

        # Ensure bag is closed on shutdown
        rospy.on_shutdown(self.shutdown_cb)

    def synced_callback(self, odom_msg):
        if not self.active:
            return
        with self.lock:
            self.bag.write('/mocap/local_position/odom', odom_msg, odom_msg.header.stamp)

    def shutdown_cb(self):
        # stop new callbacks
        self.active = False

        # unregister so ROS won’t queue any more
        self.odom_sub.unregister()

        # give any in‑flight callbacks a moment
        rospy.sleep(0.1)

        # close the bag safely
        with self.lock:
            self.bag.close()
        rospy.loginfo("RosbagRecorder: bag closed cleanly.")

if __name__ == '__main__':
    # bag_file   = sys.argv[1]
    # odom_topic = sys.argv[2]
    rospy.init_node('img_recorder_rosbag')
    bag_file   = rospy.get_param('~bag_file',   '/tmp/run.bag')

    recorder = MocapRecorder(bag_file)
    rospy.spin()
