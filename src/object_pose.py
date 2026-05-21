#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class ObjectPoseOffsetNode:
    def __init__(self):
        rospy.init_node('object_pose_offset_node')
        
        # Load parameters
        tracked_objects = rospy.get_param('~tracked_objects', 
                                        ['obstacle_1', 'obstacle_2', 'obstacle_3', 'obstacle_4', 'obstacle_5', 'obstacle_6', 'obstacle_7'])
        
        config_path = "/root/catkin_ws/src/mushr_mocap/configs/mocap_tf_offset.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
            DEG2RAD = np.pi/180.0

            self.offset_x = config["x"]
            self.offset_y = config["y"] 
            self.offset_z = config["z"] 

            self.offset_roll = config["roll"]*DEG2RAD
            self.offset_pitch = config["pitch"]*DEG2RAD
            self.offset_yaw = config["yaw"]*DEG2RAD
        
        self.publishers = {}
        
        # Create subscribers and publishers for each object
        for obj in tracked_objects:
            # Determine input topic based on whether it's the car
            input_topic = f"/vrpn_client_obstacle_poses/{obj}/pose"    
            output_topic = f"/mocap/{obj}/pose"
            
            # Create publisher
            self.publishers[obj] = rospy.Publisher(output_topic, PoseStamped, queue_size=10)
            
            # Create subscriber with callback
            rospy.Subscriber(input_topic, PoseStamped, self.pose_callback, callback_args=obj)
            
        rospy.loginfo("Pose offset node started with offsets:")
        rospy.loginfo(f"Position: [{self.offset_x}, {self.offset_y}, {self.offset_z}]")
        rospy.loginfo(f"Orientation: [{self.offset_roll}, {self.offset_pitch}, {self.offset_yaw}]")

    def pose_callback(self, msg, obj_name):
        """Process incoming pose message and apply offsets"""
        try:
            # Create new pose message
            new_pose = PoseStamped()
            new_pose.header = msg.header  # Maintain original header
            
            # Apply position offsets
            new_pose.pose.position.x = msg.pose.position.x + self.offset_x
            new_pose.pose.position.y = msg.pose.position.y + self.offset_y
            new_pose.pose.position.z = msg.pose.position.z + self.offset_z
            
            # Apply orientation offsets
            q = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            
            # Convert to Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = euler_from_quaternion(q)
            
            # Apply orientation offsets
            roll += self.offset_roll
            pitch += self.offset_pitch
            yaw += self.offset_yaw
            
            # Convert back to quaternion
            q_new = quaternion_from_euler(roll, pitch, yaw)
            
            new_pose.pose.orientation.x = q_new[0]
            new_pose.pose.orientation.y = q_new[1]
            new_pose.pose.orientation.z = q_new[2]
            new_pose.pose.orientation.w = q_new[3]
            
            # Publish transformed pose
            self.publishers[obj_name].publish(new_pose)
            
        except Exception as e:
            rospy.logerr(f"Error processing pose for {obj_name}: {str(e)}")

if __name__ == '__main__':
    try:
        node = ObjectPoseOffsetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass