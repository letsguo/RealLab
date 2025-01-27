import numpy as np
import matplotlib.pyplot as plt

MAP_RESOLUTION = 0.1

def rotate_matrix(matrix, radians):
    """
    Rotates a 2D numpy matrix counter-clockwise around its center by specified degrees.
    
    Parameters:
    matrix (numpy.ndarray): Input 2D matrix.
    degrees (float): Rotation angle in degrees (counter-clockwise).
    
    Returns:
    numpy.ndarray: Rotated matrix with the same shape as the input.
    """
    theta = radians  # Convert to radians and negate for correct direction
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, s], [-s, c]])  # Rotation matrix
    
    h, w = matrix.shape
    cy, cx = (h-1)/2.0, (w-1)/2.0  # Center coordinates
    
    # Generate grid of target coordinates
    y_out, x_out = np.indices((h, w))
    x_out_centered = x_out - cx
    y_out_centered = y_out - cy
    
    # Apply inverse rotation to find source coordinates
    x_orig_centered = rotation_matrix[0,0] * x_out_centered + rotation_matrix[0,1] * y_out_centered
    y_orig_centered = rotation_matrix[1,0] * x_out_centered + rotation_matrix[1,1] * y_out_centered
    x_orig = x_orig_centered + cx
    y_orig = y_orig_centered + cy
    
    # Bilinear interpolation
    x_floor = np.floor(x_orig).astype(int)
    y_floor = np.floor(y_orig).astype(int)
    dx = x_orig - x_floor
    dy = y_orig - y_floor
    
    # Handle out-of-bounds coordinates and clip
    x0 = np.clip(x_floor, 0, w-1)
    y0 = np.clip(y_floor, 0, h-1)
    x1 = np.clip(x_floor + 1, 0, w-1)
    y1 = np.clip(y_floor + 1, 0, h-1)
    
    # Validate coordinates (set to 0 if out of bounds)
    valid_x0 = (x_floor >= 0) & (x_floor < w)
    valid_y0 = (y_floor >= 0) & (y_floor < h)
    valid_x1 = (x_floor+1 >= 0) & (x_floor+1 < w)
    valid_y1 = (y_floor+1 >= 0) & (y_floor+1 < h)
    
    # Sample values with validity masks
    v00 = matrix[y0, x0] * (valid_x0 & valid_y0)
    v10 = matrix[y0, x1] * (valid_x1 & valid_y0)
    v01 = matrix[y1, x0] * (valid_x0 & valid_y1)
    v11 = matrix[y1, x1] * (valid_x1 & valid_y1)
    
    # Interpolation formula
    interpolated = (
        (v00 * (1 - dx) + v10 * dx) * (1 - dy) +
        (v01 * (1 - dx) + v11 * dx) * dy
    )
    
    return interpolated.astype(matrix.dtype)

def crop_heightmap(heightmap, x, y, yaw, width=20):
    angle_rad = -yaw
    heightmap = rotate_matrix(heightmap, angle_rad)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta
    rows, cols = heightmap.shape
    j = int((new_x / MAP_RESOLUTION) + cols//2)
    i = int((new_y / MAP_RESOLUTION) + rows//2)
    cropped = np.zeros((width, width))
    for i_c in range(width):
        for j_c in range(width):
            i_h = i - width//2 + i_c
            j_h = j - width//2 + j_c
            if 0 <= i_h < rows and 0 <= j_h < cols:
                cropped[i_c, j_c] = heightmap[i_h, j_h]
    return cropped

def generate_heightmap(obstacles, heightmap, block, ramp):
    def overlay_heightmaps(large, small, i, j):
        # Create a deep copy of the large heightmap to avoid modifying the original
        new_large = np.copy(large)
        
        # Get dimensions of the small heightmap
        S_rows = len(small)
        if S_rows == 0:
            return new_large  # Nothing to overlay
        S_cols = len(small[0])
        if S_cols == 0:
            return new_large  # Nothing to overlay
        
        # Get dimensions of the large heightmap
        L_rows = len(new_large)
        if L_rows == 0:
            return new_large  # No large map to overlay onto
        L_cols = len(new_large[0]) if L_rows > 0 else 0
        
        # Calculate the center position of the small heightmap
        S_center_row = (S_rows - 1) // 2
        S_center_col = (S_cols - 1) // 2
        
        # Iterate over each element in the small heightmap
        for s_r in range(S_rows):
            for s_c in range(S_cols):
                # Calculate corresponding position in the large heightmap
                l_r = i - S_center_row + s_r
                l_c = j - S_center_col + s_c
                
                # Check if the position is within the bounds of the large heightmap
                if 0 <= l_r < L_rows and 0 <= l_c < L_cols:
                    new_large[l_r][l_c] += small[s_r][s_c]
        
        return new_large
    
    def xy_to_ij(x, y, heightmap):
        rows, cols = heightmap.shape
        y = y / MAP_RESOLUTION + rows // 2
        x = x / MAP_RESOLUTION + cols // 2
        return int(y), int(x)
    
    for obstacle in obstacles:
        x_obs = obstacle['position'][0]
        y_obs = obstacle['position'][1]
        yaw = -obstacle['orientation']
        obstacle = block if obstacle['type'] == 'block' else ramp
        obstacle = rotate_matrix(obstacle, yaw)
        i, j = xy_to_ij(x_obs, y_obs, heightmap)
        heightmap = overlay_heightmaps(heightmap, obstacle, i, j)
        
    return heightmap