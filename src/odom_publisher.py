#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_inverse, quaternion_multiply, quaternion_matrix, unit_vector
import numpy as np

class OdometryProcessorNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('odometry_processor', anonymous=True)

        # Subscribers
        self.gt_odom_sub = rospy.Subscriber('/camera/odom/sample', Odometry, self.relative_callback)
        self.rel_odom_sub = rospy.Subscriber('/mocap/local_position/odom', Odometry, self.gt_callback)

        # Publisher
        self.fused_odom_pub = rospy.Publisher('/mavros/local_position/odom', Odometry, queue_size=1)  # Update message type if needed
        self.fused_pose_pub = rospy.Publisher('/mavros/local_position/pose', PoseStamped, queue_size=1)

        # set these to whatever you want starting position to be
        self.absolute_quat = [0.0,0.0,0.707,0.707]
        
        # for drift
        # self.xyz_offsets = np.array([1.0,0.0,0.0])
        # for mppi
        self.xyz_offsets = np.array([0.0,0.0,0.0])

        
        self.offset_quat = [0.0,0.0,0.0,1.0]
        self.cam_rotation_quat = [1.0,0.0,0.0,0.0]
        self.rel_init = [0.0,0.0,0.0,1.0]
        self.relative_angle_init = [0.0,0.0,0.0,1.0]
        self.absolute_angle_init = self.absolute_quat
        self.relative_raw = [0.0,0.0,0.0,1.0]
        self.relative_quat = [0.0,0.0,0.0,1.0]
        self.odom = Odometry()
        self.odom.header.frame_id = "map"
        self.odom.child_frame_id = "base_link"
        self.odom_update = False
        self.offsets_init = False

        self.relative_xyz = np.array([0.0,0.0,0.0])        

        rospy.loginfo('Odometry Processor Node started')

        # Run the processing loop
        self.rate = rospy.Rate(100)  # 10 Hz
        self.main_loop()

    def rotate_vector(self, vector, quaternion):
        # Convert quaternion to a rotation matrix
        rotation_matrix = quaternion_matrix(quaternion)[:3, :3]
        
        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector
    
    def rotate_quaternion(self, quaternion, transformation):
        return quaternion_multiply(
            quaternion_multiply(transformation, quaternion),
            quaternion_inverse(transformation)
        )

    def gt_callback(self, msg):
        self.absolute_quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        self.rel_init = self.relative_raw
        relative_quat = quaternion_multiply(quaternion_inverse(self.relative_angle_init), self.relative_raw)
        rel = quaternion_multiply(self.absolute_angle_init, relative_quat)
        rel_inverse = quaternion_inverse(rel)
        self.offset_quat = quaternion_multiply(self.absolute_angle_init, 
                                    quaternion_multiply(rel_inverse, self.absolute_quat))
        xyz_rel = self.rotate_vector(self.relative_xyz, self.offset_quat)
        xyz_gt = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.xyz_offsets = xyz_gt - xyz_rel

    def set_offsets(self):
        self.relative_angle_init = self.rel_init
        self.offset_quat = self.absolute_quat

    def relative_callback(self, msg):
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        self.relative_raw = self.rotate_quaternion(quat, self.cam_rotation_quat)
        if not self.offsets_init:
            self.rel_init = self.relative_raw

        self.relative_quat = quaternion_multiply(quaternion_inverse(self.rel_init), self.relative_raw)

        relative_xyz = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.relative_xyz = self.rotate_vector(relative_xyz, [0.0,0.0,1.0,0.0])

        if not self.offsets_init:
            self.set_offsets()
            self.offsets_init = True

        quat = quaternion_multiply(self.absolute_quat, self.relative_quat)
        xyz = self.rotate_vector(self.relative_xyz, self.offset_quat) + self.xyz_offsets
        self.odom.pose.pose.orientation.x = quat[0]
        self.odom.pose.pose.orientation.y = quat[1]
        self.odom.pose.pose.orientation.z = quat[2]
        self.odom.pose.pose.orientation.w = quat[3]

        self.odom.pose.pose.position.x = xyz[0]
        self.odom.pose.pose.position.y = xyz[1]
        self.odom.pose.pose.position.z = xyz[2]

        xyz_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z            
        ])
        xyz_vel = self.rotate_vector(xyz_vel, self.cam_rotation_quat)
        self.odom.twist.twist.linear.x = xyz_vel[0]
        self.odom.twist.twist.linear.y = xyz_vel[1]
        self.odom.twist.twist.linear.z = xyz_vel[2]

        xyz_angular_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z            
        ])
        xyz_angular_vel = self.rotate_vector(xyz_angular_vel, self.cam_rotation_quat)
        self.odom.twist.twist.angular.x = xyz_angular_vel[0]
        self.odom.twist.twist.angular.y = xyz_angular_vel[1]
        self.odom.twist.twist.angular.z = xyz_angular_vel[2]

        self.odom.header.stamp = msg.header.stamp
        self.odom_update = True

    def main_loop(self):
        while not rospy.is_shutdown():
            if self.odom_update:
                self.fused_odom_pub.publish(self.odom)
                pose = PoseStamped()
                pose.header = self.odom.header
                pose.pose.position = self.odom.pose.pose.position
                pose.pose.orientation = self.odom.pose.pose.orientation
                self.fused_pose_pub.publish(pose)
                self.odom_update = False
                self.rate.sleep()

if __name__ == '__main__':
    try:
        node = OdometryProcessorNode()
    except rospy.ROSInterruptException:
        rospy.loginfo('Odometry Processor Node shutting down')