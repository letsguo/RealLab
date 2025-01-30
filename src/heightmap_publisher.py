#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import RCIn
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils.generate_elevation_map import generate_heightmap  # Import your actual function

class BlockHeightmapGenerator:
    def __init__(self):
        rospy.init_node('block_heightmap_generator')
        
        # Load parameters
        tracked_objects = rospy.get_param('~tracked_objects', 
                                        ['block1', 'block2', 'block3', 'block4', 'ramp1', 'ramp2'])
        self.rate = 1  # Processing rate in Hz
        
        # Initialize data stores
        self.obstacles = {}

        # Setup tracked objects
        self.rc_sub = rospy.Subscriber('/car/teleop/joy', RCIn, self.rcin_callback)
        self.car_off = True

        for obj in tracked_objects:
            # Create publisher for corrected pose
            input_topic = f"/mocap/{obj}/pose"
            
            # Create subscriber for raw pose
            rospy.Subscriber(input_topic, PoseStamped, 
                           self.pose_callback, callback_args=obj)
            
            # Initialize block storage for block objects
            if obj.startswith('block'):
                self.obstacles[obj] = None

        self.map_pub = rospy.Publisher('heightmap', Float32MultiArray, queue_size=1)

        obstacle_list = [b for b in self.obstacles.values() if b is not None]
        self.heightmap_raw = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap.npy")
        self.block = np.load('/root/catkin_ws/src/hound_core/config/elevation/block.npy')
        self.ramp = np.load('/root/catkin_ws/src/hound_core/config/elevation/ramp.npy') 

        self.heightmap = generate_heightmap(obstacle_list, self.heightmap_raw, self.block, self.ramp)

        self.main_loop()

    def main_loop(self):
        """Main loop to update heightmap"""
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.car_off:
                self.update_heightmap()
            rate.sleep()

    def rcin_callback(self, data):
        try:
            self.car_off = data.buttons[5] == 0        
        except Exception as e:
            pass

    def pose_callback(self, msg, obj_name):
        """Process raw pose data, apply offsets, and store block information"""
        
        # Apply orientation offsets
        orig_q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        _, _, yaw = euler_from_quaternion(orig_q)
        
        # Store block data if applicable
        if obj_name.startswith('block'):
            self.obstacles[obj_name] = {
                'type': 'block',
                'position': (msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z),
                'orientation': yaw
            }
        elif obj_name.startswith('ramp'):
            self.obstacles[obj_name] = {
                'type': 'ramp',
                'position': (msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z),
                'orientation': yaw
            }

    def update_heightmap(self):
        """Generate heightmap from block positions"""
        # Prepare block data
        obstacle_list = [b for b in self.obstacles.values() if b is not None]
        
        self.heightmap = generate_heightmap(obstacle_list, self.heightmap_raw, self.block, self.ramp)
        msg = Float32MultiArray()
        msg.data = self.heightmap.flatten().tolist()
        self.map_pub.publish(msg)

if __name__ == '__main__':
    try:
        BlockHeightmapGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass