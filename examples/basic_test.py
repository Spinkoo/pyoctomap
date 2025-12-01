import open3d as o3d
import numpy as np

# --- Log-Odds Helper Functions (unchanged) ---
def prob_to_log_odds(p):
    return np.log(p / (1 - p))

def log_odds_to_prob(l):
    return 1 - (1 / (1 + np.exp(l)))

# --- Configuration (unchanged) ---
SENSOR_ORIGIN = np.array([0, 0, 0])
LOG_ODDS_OCCUPIED = prob_to_log_odds(0.85)
LOG_ODDS_FREE = prob_to_log_odds(0.40)

# --- Main Script ---ore  e sophist
if __name__ == "__main__":
    # 1. Create a sample point cloud (unchanged)
    point_cloud_np = (np.random.rand(200, 3) - 0.5) * 10
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    
    # 2. Define the space with an Octree using Open3D's built-in functionality
    # Create octree from point cloud
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(pcd, size_expand=0.1)
    
    # 3. Initialize the Occupancy Map (unchanged)
    log_odds_map = {}

    # 4. Process each point (ray casting and log-odds update)
    print("Processing sensor readings...")
    for point in point_cloud_np:
        ray_vector = point - SENSOR_ORIGIN
        ray_length = np.linalg.norm(ray_vector)
        step_size = 0.1  # Use a fixed step size for simplicity
        
        # Ray casting through the octree
        for i in np.arange(0, ray_length, step_size):
            pos_on_ray = SENSOR_ORIGIN + ray_vector * (i / ray_length)
            
            # Find the leaf node containing this position
            # Since Open3D's octree doesn't have locate_leaf_node, we'll use a different approach
            # We'll create a simple grid-based occupancy map instead
            
            # Discretize the position to grid coordinates
            grid_pos = np.floor(pos_on_ray / step_size).astype(int)
            grid_key = tuple(grid_pos)
            
            current_log_odds = log_odds_map.get(grid_key, 0)
            log_odds_map[grid_key] = current_log_odds - LOG_ODDS_FREE

        # Mark the endpoint as occupied
        grid_pos = np.floor(point / step_size).astype(int)
        grid_key = tuple(grid_pos)
        current_log_odds = log_odds_map.get(grid_key, 0)
        log_odds_map[grid_key] = current_log_odds + LOG_ODDS_OCCUPIED

    print(f"Map created with {len(log_odds_map)} updated cells.")

    # 5. Visualize the Probabilistic OGM
    ogm_points = []
    ogm_colors = []
    for grid_key, log_odds_val in log_odds_map.items():
        prob = log_odds_to_prob(log_odds_val)
        grid_pos = np.array(grid_key)
        node_center = grid_pos * step_size + step_size / 2.0
        
        ogm_points.append(node_center)
        if prob > 0.7:
            ogm_colors.append([1.0, 0.0, 0.0]) # Red
        elif prob < 0.3:
            ogm_colors.append([0.0, 0.0, 1.0]) # Blue
        else:
            ogm_colors.append([0.5, 0.5, 0.5]) # Gray
            
    ogm_pcd = o3d.geometry.PointCloud()
    ogm_points = np.array(ogm_points)
    if len(ogm_points) > 0:
        ogm_pcd.points = o3d.utility.Vector3dVector(ogm_points)
        ogm_colors = np.array(ogm_colors)
        ogm_pcd.colors = o3d.utility.Vector3dVector(ogm_colors)

    pcd.paint_uniform_color([1.0, 1.0, 1.0]) 

    print("Visualizing OGM: Red=Occupied, Blue=Free, Gray=Uncertain, White=Original Points")
    
    # Create a simple visualization
    geometries = [pcd]
    if len(ogm_points) > 0:
        geometries.append(ogm_pcd)
    
    o3d.visualization.draw_geometries(geometries)