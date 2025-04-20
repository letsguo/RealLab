#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

class Teleop:
    def __init__(self):
        # Axis config for PS5 sticks
        # self.throttle_axis = rospy.get_param("~throttle_axis", 1)   # Left stick Y
        # self.steering_axis = rospy.get_param("~steering_axis", 2)   # Right stick X
        self.throttle_axis = 1
        self.steering_axis = 3

        # self.max_speed = rospy.get_param("~max_speed", 2.0)               # m/s
        # self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)  # radians
        self.max_speed = 2.0
        self.max_steering_angle = 0.34

        # Publisher
        self.pub = rospy.Publisher("/car/mux/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

        # Subscriber
        rospy.Subscriber("/car/teleop/joy", Joy, self.joy_callback)

    def joy_callback(self, msg):
        drive_msg = AckermannDriveStamped()
        drive = drive_msg.drive

        # Throttle: invert Y-axis so pushing forward is positive
        throttle_input = msg.axes[self.throttle_axis]
        drive.speed = throttle_input * self.max_speed

        # Steering: right stick X-axis
        steer_input = msg.axes[self.steering_axis]
        drive.steering_angle = -steer_input * self.max_steering_angle

        self.pub.publish(drive_msg)


if __name__ == "__main__":
    rospy.init_node("teleop")
    # planner = Image_Recorder('/car/vesc/odom')
    teleop = Teleop()
    rospy.spin()