#!/usr/bin/env python3
import rospy
import threading
import numpy as np
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables for data sharing
latest_data = None
data_lock = threading.Lock()

def hl_state_callback(msg):
    global latest_data
    # Extract and reshape heightmap data
    heightmap_flat = msg.data[14:690]
    with data_lock:
        latest_data = np.array(heightmap_flat).reshape((26, 26))

def update_plot(frame):
    global latest_data
    with data_lock:
        current_data = latest_data.copy() if latest_data is not None else None
    
    if current_data is not None:
        ax.clear()
        # Create 2D heatmap
        img = ax.imshow(current_data, cmap='viridis', origin='lower')
        ax.set_title("2D Height Map Visualization")
        
        # Add colorbar (only once)
        if not hasattr(update_plot, 'cbar'):
            update_plot.cbar = fig.colorbar(img, ax=ax)
            update_plot.cbar.set_label('Elevation')

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
    
            
    return [img]

if __name__ == '__main__':
    rospy.init_node('heightmap_2d_visualizer')
    rospy.Subscriber('hl_controller/state', Float32MultiArray, hl_state_callback)

    rospy.sleep(1)

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    
    # Initial empty plot
    img = ax.imshow(np.zeros((26,26)), cmap='viridis', origin='lower', 
                   vmin=0, vmax=1)  # Adjust vmin/vmax based on your data range
    
    # Start ROS spinner in background
    spinner = threading.Thread(target=rospy.spin)
    spinner.daemon = True
    spinner.start()
    
    # Set up animation at 5Hz (200ms interval)
    ani = FuncAnimation(fig, update_plot, interval=5, blit=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Shutting down visualization...")