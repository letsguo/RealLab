#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Float32MultiArray

class StateSplitterNode:
    def __init__(self):
        rospy.init_node('state_splitter', anonymous=True)

        # Subscription to the input topic
        self.subscription = rospy.Subscriber(
            '/hl_controller/state',
            Float32MultiArray,
            self.state_callback
        )

        self.publishers = []
        rospy.loginfo('State Splitter Node initialized')

    def state_callback(self, msg):
        # Ensure there are enough publishers
        while len(self.publishers) < len(msg.data):
            topic_name = f'/state{len(self.publishers) + 1}'
            publisher = rospy.Publisher(topic_name, Float32, queue_size=10)
            self.publishers.append(publisher)

        # Publish each item in the array to its respective topic
        for i, value in enumerate(msg.data):
            float_msg = Float32()
            float_msg.data = value
            self.publishers[i].publish(float_msg)

def main():
    node = StateSplitterNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down State Splitter Node')

if __name__ == '__main__':
    main()
