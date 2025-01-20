#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped

class AckermannToSteeringThrottleNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('ackermann_to_steering_throttle_node', anonymous=True)

        # Subscriber to the control topic (AckermannDriveStamped)
        self.subscription = rospy.Subscriber(
            '/low_level_controller/hound/control',
            AckermannDriveStamped,
            self.listener_callback
        )

        # Publishers for steering and throttle
        self.steering_pub = rospy.Publisher('/steering', Float32, queue_size=10)
        self.throttle_pub = rospy.Publisher('/throttle', Float32, queue_size=10)

    def listener_callback(self, msg):
        steering_msg = Float32()
        throttle_msg = Float32()

        # Extract steering and throttle from AckermannDriveStamped message
        steering_msg.data = msg.drive.steering_angle
        throttle_msg.data = msg.drive.speed
        
        # Publish the steering and throttle
        self.steering_pub.publish(steering_msg)
        self.throttle_pub.publish(throttle_msg)

    def spin(self):
        # Keep the node running and handle callbacks
        rospy.spin()

if __name__ == '__main__':
    node = AckermannToSteeringThrottleNode()
    node.spin()
