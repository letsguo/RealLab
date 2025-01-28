#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from datetime import datetime

class ArrayCollectorNode:
    def __init__(self):
        rospy.init_node('array_collector_node', anonymous=True)
        
        # List to accumulate all received arrays
        self.array_list = []
        
        # Create subscriber with queue size 10
        self.sub = rospy.Subscriber('value', Float32MultiArray,
                                   self.callback, queue_size=10)
        
        # Register shutdown hook to save when node stops
        rospy.on_shutdown(self.save_all_arrays)
        
        rospy.loginfo("Array Collector Node initialized. Collecting data...")

    def callback(self, msg):
        # Validate and extract first 14 elements
        if len(msg.data) < 15:
            rospy.logerr("Received array with less than 14 elements!")
            return
            
        data = msg.data[:15]
        
        try:
            # Convert to numpy array and store in list
            array = np.array(data, dtype=np.float32)
            self.array_list.append(array)            
        except Exception as e:
            rospy.logerr(f"Error processing data: {str(e)}")

    def save_all_arrays(self):
        if not self.array_list:
            rospy.logwarn("No arrays collected, nothing to save!")
            return
            
        try:
            # Stack all arrays into a single 2D array
            combined_array = np.stack(self.array_list, axis=0)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/root/catkin_ws/src/hound_core/data/combined_arrays_{timestamp}.npy"
            
            # Save the combined array
            np.save(filename, combined_array)
            rospy.loginfo(f"Saved {len(self.array_list)} arrays to {filename}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save combined array: {str(e)}")

if __name__ == '__main__':
    try:
        node = ArrayCollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass