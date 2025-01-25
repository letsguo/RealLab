import rosbag
import torch
import numpy as np

# Function to convert ROS bag topic data into PyTorch tensors
def rosbag_to_tensors(bag_file, topics):
    """
    Converts data from specified topics in a ROS bag file into PyTorch tensors.

    Args:
        bag_file (str): Path to the ROS bag file.
        topics (list of str): List of topic names to extract.

    Returns:
        dict: A dictionary where keys are topic names and values are PyTorch tensors.
    """
    bag = rosbag.Bag(bag_file, 'r')
    topic_data = {topic: [] for topic in topics}

    # Read messages from the bag file
    for topic, msg, t in bag.read_messages(topics=topics):
        if hasattr(msg, 'pose'):
            # Example: extracting pose data from geometry_msgs/PoseStamped
            pose = msg.pose.pose
            print(pose.position.x + " " + pose.position.y)
            data = np.array([
                pose.position.x, pose.position.y])
            topic_data[topic].append(data)
        else:
            print(f"Unsupported message type for topic: {topic}")

    bag.close()

    tensor = None

    # Convert collected data to PyTorch tensors
    for topic, values in topic_data.items():
        tensor = torch.tensor(np.array(values), dtype=torch.float32)

    return tensor

# Example usage
if __name__ == "__main__":
    bag_file = "example.bag"  # Replace with the path to your ROS bag file
    topics = ["/topic1", "/topic2"]  # Replace with the list of topics you want to extract

    tensor = rosbag_to_tensors(bag_file, topics)

    print(f"Tensor Shape: {tensor.shape}")

    # # Print tensor shapes
    # for topic, tensor in tensors.items():
    #     print(f"Topic: {topic}, Tensor Shape: {tensor.shape}")