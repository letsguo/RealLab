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

        self.img_sub  = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.odom_sub = message_filters.Subscriber('/car/odom', Odometry)
        self.ang_vel_sub = message_filters.Subscriber('/camera/gyro/sample', Imu)

        ats = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.odom_sub, self.ang_vel_sub],
            queue_size=10,
            slop=0.05,  # allow 50ms time difference
            allow_headerless=False
        )

        ats.registerCallback(self.synced_callback)
        # Ensure bag is closed on shutdown
        rospy.on_shutdown(self.shutdown_cb)

    def synced_callback(self, image_msg, odom_msg, ang_vel_msg):
        with self.lock:
            self.bag.write('/camera/color/image_raw', image_msg, image_msg.header.stamp)
            self.bag.write('/car/odom', odom_msg, odom_msg.header.stamp)
            self.bag.write('/camera/gyro/sample', ang_vel_msg, ang_vel_msg.header.stamp)

    def shutdown_cb(self):
        # stop new callbacks
        self.active = False

        # unregister so ROS won’t queue any more
        self.img_sub.unregister()
        self.odom_sub.unregister()
        self.ang_vel_sub.unregister()

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

    recorder = RosbagRecorder(bag_file)
    rospy.spin()
