#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

class OdomProcessor:
    def __init__(self):
        # Initialize the node
        rospy.init_node('odom_processor', anonymous=True)

        # Publishers for pose
        self.pose_x_pub = rospy.Publisher('/pose_x', Float32, queue_size=10)
        self.pose_y_pub = rospy.Publisher('/pose_y', Float32, queue_size=10)
        self.pose_z_pub = rospy.Publisher('/pose_z', Float32, queue_size=10)

        # Publishers for linear velocity
        self.vel_x_pub = rospy.Publisher('/velocity_x', Float32, queue_size=10)
        self.vel_y_pub = rospy.Publisher('/velocity_y', Float32, queue_size=10)
        self.vel_z_pub = rospy.Publisher('/velocity_z', Float32, queue_size=10)

        # Publishers for orientation (quaternion)
        self.orientation_x_pub = rospy.Publisher('/orientation_x', Float32, queue_size=10)
        self.orientation_y_pub = rospy.Publisher('/orientation_y', Float32, queue_size=10)
        self.orientation_z_pub = rospy.Publisher('/orientation_z', Float32, queue_size=10)
        self.orientation_w_pub = rospy.Publisher('/orientation_w', Float32, queue_size=10)

        # Subscribers
        self.subscription = rospy.Subscriber(
            '/mavros/local_position/odom',
            Odometry,
            self.odom_callback
        )

    def odom_callback(self, msg):
        # Extract pose
        pose_x = Float32()
        pose_x.data = msg.pose.pose.position.x
        self.pose_x_pub.publish(pose_x)

        pose_y = Float32()
        pose_y.data = msg.pose.pose.position.y
        self.pose_y_pub.publish(pose_y)

        pose_z = Float32()
        pose_z.data = msg.pose.pose.position.z
        self.pose_z_pub.publish(pose_z)

        # Extract linear velocity
        vel_x = Float32()
        vel_x.data = msg.twist.twist.linear.x
        self.vel_x_pub.publish(vel_x)

        vel_y = Float32()
        vel_y.data = msg.twist.twist.linear.y
        self.vel_y_pub.publish(vel_y)

        vel_z = Float32()
        vel_z.data = msg.twist.twist.linear.z
        self.vel_z_pub.publish(vel_z)

        # Extract orientation (quaternion)
        orientation_x = Float32()
        orientation_x.data = msg.pose.pose.orientation.x
        self.orientation_x_pub.publish(orientation_x)

        orientation_y = Float32()
        orientation_y.data = msg.pose.pose.orientation.y
        self.orientation_y_pub.publish(orientation_y)

        orientation_z = Float32()
        orientation_z.data = msg.pose.pose.orientation.z
        self.orientation_z_pub.publish(orientation_z)

        orientation_w = Float32()
        orientation_w.data = msg.pose.pose.orientation.w
        self.orientation_w_pub.publish(orientation_w)


def main():
    processor = OdomProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
