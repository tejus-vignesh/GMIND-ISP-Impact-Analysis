import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Calibration', 'camera_intrinsics')))
from camera import CameraModel

def parse_sensor_calibration(calib_path):
    """
    Parse sensor_calibration.txt file and return camera objects and extrinsics.
    Returns: (cameras dict, extrinsics dict, image_dimensions dict)
    """
    cameras = {}
    extrinsics = {}
    image_dimensions = {}
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Name:'):
            name = line.split(':', 1)[1].strip()
            
            # Read intrinsics (if present)
            intrinsics = {}
            while i < len(lines) and not lines[i].strip().startswith('Extrinsics'):
                l = lines[i].strip()
                if ':' in l:
                    k, v = l.split(':', 1)
                    intrinsics[k.strip()] = v.strip()
                i += 1
            
            # Create camera matrix and distortion (only for cameras with intrinsics)
            if 'Focal_x' in intrinsics:
                fx = float(intrinsics.get('Focal_x', 0))
                fy = float(intrinsics.get('Focal_y', 0))
                cx = float(intrinsics.get('COD_x', 0))
                cy = float(intrinsics.get('COD_y', 0))
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = [float(intrinsics.get(f'Dist_{j+1}', 0)) for j in range(4)]
                cameras[name] = CameraModel(camera_matrix, dist_coeffs)
                
                # Store image dimensions if available
                width = intrinsics.get('Width', '')
                height = intrinsics.get('Height', '')
                if width and height:
                    try:
                        image_dimensions[name] = (int(width), int(height))
                    except ValueError:
                        pass
            
            # Read extrinsics
            extr = {'X': 0, 'Y': 0, 'Z': 0, 'R': np.eye(3, dtype=np.float32)}
            while i < len(lines) and lines[i].strip() != '':
                l = lines[i].strip()
                if ':' in l:
                    k, v = l.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ['X', 'Y', 'Z']:
                        try:
                            extr[k] = float(v) if v else 0.0
                        except ValueError:
                            extr[k] = 0.0
                    elif k.startswith('R_'):
                        idx = k[2:]
                        if len(idx) >= 2:
                            row, col = int(idx[0]), int(idx[1])
                            if 'R' not in extr or not isinstance(extr['R'], np.ndarray):
                                extr['R'] = np.eye(3, dtype=np.float32)
                            try:
                                extr['R'][row, col] = float(v)
                            except ValueError:
                                extr['R'][row, col] = 0
                i += 1
            
            if name in cameras:  # Only store extrinsics for cameras
                extrinsics[name] = extr
        i += 1
    
    return cameras, extrinsics, image_dimensions

def transform_ray_to_camera_coords(ray_cam1, cam1_extr, cam2_extr):
    """
    Transform a ray (direction vector) from camera1 coordinate system to camera2 coordinate system.
    
    Args:
        ray_cam1: Normalized direction vector in camera1 frame (3,)
        cam1_extr: Extrinsics dict for camera1 with 'R' (rotation) and 'X', 'Y', 'Z' (translation)
        cam2_extr: Extrinsics dict for camera2
    
    Returns:
        ray_cam2: Normalized direction vector in camera2 frame (3,)
    """
    # Transform ray from camera1 to world frame
    # In world frame, the ray starts at cam1 position and goes in direction R_cam1 @ ray_cam1
    R_cam1 = cam1_extr['R']
    t_cam1 = np.array([cam1_extr['X'], cam1_extr['Y'], cam1_extr['Z']], dtype=np.float32)
    
    # Ray direction in world frame
    ray_world = R_cam1 @ ray_cam1
    
    # Transform from world frame to camera2 frame
    # Inverse transformation: R_cam2.T @ (point_world - t_cam2)
    # For a direction vector (no translation), it's just R_cam2.T @ ray_world
    R_cam2 = cam2_extr['R']
    t_cam2 = np.array([cam2_extr['X'], cam2_extr['Y'], cam2_extr['Z']], dtype=np.float32)
    
    ray_cam2 = R_cam2.T @ ray_world
    # Normalize to ensure it's a unit vector
    ray_cam2 = ray_cam2 / np.linalg.norm(ray_cam2)
    
    return ray_cam2

def project_ray_at_infinity(camera_model, ray_cam, distance=1e6):
    """
    Project a ray (direction vector) at infinity onto the camera image.
    Uses a very large distance to approximate infinity.
    
    Args:
        camera_model: CameraModel object
        ray_cam: Normalized direction vector in camera frame (3,)
        distance: Large distance to use for projection (default: 1e6 meters)
    
    Returns:
        pixel: 2D pixel coordinates (u, v) or None if behind camera
    """
    # Check if ray is pointing away from camera (negative Z)
    if ray_cam[2] <= 0:
        return None
    
    # Create a 3D point along the ray at the given distance
    point_3d = ray_cam * distance
    point_3d = point_3d.reshape(1, 3)
    
    # Project to 2D
    pixels = camera_model.project_point(point_3d)
    return pixels[0]

def transform_point_to_camera_coords(point_cam1, cam1_extr, cam2_extr):
    """
    Transform a 3D point from camera1 coordinate system to camera2 coordinate system.
    
    Args:
        point_cam1: 3D point in camera1 frame (3,)
        cam1_extr: Extrinsics dict for camera1
        cam2_extr: Extrinsics dict for camera2
    
    Returns:
        point_cam2: 3D point in camera2 frame (3,)
    """
    # Transform point from camera1 to world frame
    R_cam1 = cam1_extr['R']
    t_cam1 = np.array([cam1_extr['X'], cam1_extr['Y'], cam1_extr['Z']], dtype=np.float32)
    point_world = R_cam1 @ point_cam1 + t_cam1
    
    # Transform from world frame to camera2 frame
    R_cam2 = cam2_extr['R']
    t_cam2 = np.array([cam2_extr['X'], cam2_extr['Y'], cam2_extr['Z']], dtype=np.float32)
    point_cam2 = R_cam2.T @ (point_world - t_cam2)
    
    return point_cam2

def create_grid_points(width, height, grid_size=20, crop_outer_percent=30):
    """
    Create evenly spaced grid points in an image, excluding the outer area.
    
    Args:
        width: Image width
        height: Image height
        grid_size: Number of points per dimension (grid_size x grid_size)
        crop_outer_percent: Percentage of outer area to exclude (default: 30%)
                           This means using the inner (100 - crop_outer_percent)% of the image
    
    Returns:
        points: Array of (u, v) pixel coordinates (grid_size*grid_size, 2)
    """
    # Calculate inner region boundaries (exclude outer crop_outer_percent/2 on each side)
    crop_factor = crop_outer_percent / 100.0
    u_min = width * crop_factor / 2.0
    u_max = width * (1 - crop_factor / 2.0)
    v_min = height * crop_factor / 2.0
    v_max = height * (1 - crop_factor / 2.0)
    
    # Create evenly spaced grid within the inner region
    u_coords = np.linspace(u_min, u_max - 1, grid_size)
    v_coords = np.linspace(v_min, v_max - 1, grid_size)
    
    # Create meshgrid and flatten
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    points = np.stack([u_grid.flatten(), v_grid.flatten()], axis=1)
    
    return points

def calculate_infinity_error(flir_camera, cctv_camera, flir_extr, cctv_extr, 
                             flir_width, flir_height, grid_size=20, test_distances=[4, 100]):
    """
    Calculate the error (delta) caused by assuming objects are at infinity.
    
    For each grid point in FLIR at a given distance:
    1. Project using infinity assumption: get direction ray, transform to CCTV, project with large distance
    2. Project using actual distance: get 3D point at actual distance, transform to CCTV, project
    3. Calculate pixel error (delta) between the two projections
    
    Args:
        flir_camera: CameraModel for FLIR 8.9MP
        cctv_camera: CameraModel for CCTV
        flir_extr: Extrinsics dict for FLIR
        cctv_extr: Extrinsics dict for CCTV
        flir_width: FLIR image width
        flir_height: FLIR image height
        grid_size: Grid size (grid_size x grid_size points)
        test_distances: List of finite distances to test (typically [4, 100] for close and far)
    
    Returns:
        results: Dict with error statistics for each test distance
    """
    # Create grid points in FLIR image
    grid_points_flir = create_grid_points(flir_width, flir_height, grid_size)
    print(f"Created {len(grid_points_flir)} grid points in FLIR image")
    
    # Convert pixels to rays in FLIR camera frame
    rays_flir = flir_camera.pixel_to_ray(grid_points_flir)  # (N, 3)
    print(f"Converted {len(rays_flir)} pixels to rays")
    
    # Transform rays from FLIR to CCTV camera frame (for infinity assumption)
    rays_cctv_infinity = np.array([transform_ray_to_camera_coords(ray, flir_extr, cctv_extr) 
                                    for ray in rays_flir])
    
    # Project rays at infinity to CCTV image
    infinity_pixels = []
    for ray in rays_cctv_infinity:
        if ray[2] > 0:  # Only project if pointing forward
            pixel = project_ray_at_infinity(cctv_camera, ray)
            infinity_pixels.append(pixel)
        else:
            infinity_pixels.append(None)
    
    infinity_pixels = np.array([p if p is not None else [np.nan, np.nan] for p in infinity_pixels])
    
    # Test at different finite distances
    results = {}
    
    for distance in test_distances:
        print(f"\nTesting at distance: {distance}m")
        
        # Create 3D points in FLIR camera frame at the specified distance
        points_flir_3d = rays_flir * distance  # (N, 3)
        
        # Transform points from FLIR to CCTV camera frame
        points_cctv_3d = np.array([transform_point_to_camera_coords(p, flir_extr, cctv_extr) 
                                    for p in points_flir_3d])
        
        # Project to CCTV image
        finite_pixels = []
        for i, point in enumerate(points_cctv_3d):
            if point[2] > 0:  # Only project if in front of camera
                pixel = cctv_camera.project_point(point.reshape(1, 3))[0]
                finite_pixels.append(pixel)
            else:
                finite_pixels.append([np.nan, np.nan])
        
        finite_pixels = np.array(finite_pixels)
        
        # Calculate error delta (pixel distance between infinity and actual distance projections)
        valid_mask = ~(np.isnan(infinity_pixels).any(axis=1) | np.isnan(finite_pixels).any(axis=1))
        if np.sum(valid_mask) > 0:
            # Calculate pixel error (delta) as the distance between infinity and actual projections
            pixel_deltas = finite_pixels[valid_mask] - infinity_pixels[valid_mask]
            errors = np.linalg.norm(pixel_deltas, axis=1)
            
            results[distance] = {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors),
                'std_error': np.std(errors),
                'valid_points': np.sum(valid_mask),
                'total_points': len(grid_points_flir),
                'errors': errors,
                'pixel_deltas': pixel_deltas,  # (u, v) deltas for each point
                'infinity_pixels': infinity_pixels[valid_mask],
                'finite_pixels': finite_pixels[valid_mask],
                'valid_mask': valid_mask  # Store which indices are valid for this distance
            }
            
            print(f"  Valid points: {results[distance]['valid_points']}/{results[distance]['total_points']}")
            print(f"  Mean error (delta): {results[distance]['mean_error']:.3f} pixels")
            print(f"  Median error (delta): {results[distance]['median_error']:.3f} pixels")
            print(f"  Max error (delta): {results[distance]['max_error']:.3f} pixels")
            print(f"  Std error (delta): {results[distance]['std_error']:.3f} pixels")
        else:
            print(f"  No valid points at distance {distance}m")
            results[distance] = None
    
    return results, grid_points_flir, rays_flir, infinity_pixels

def plot_debug_results(results, grid_points_flir, infinity_pixels, flir_width, flir_height, 
                       cctv_width, cctv_height, output_path='infinity_error_debug.png'):
    """
    Create debug plots showing FLIR grid, CCTV projections at infinity vs actual distances,
    and error vectors for 4m, 20m, and 100m.
    """
    distances = sorted([d for d in results.keys() if results[d] is not None])
    
    if len(distances) == 0:
        print("No valid results to plot")
        return None
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout: one row per distance, showing FLIR grid, CCTV infinity, CCTV finite, and error vectors
    n_distances = len(distances)
    
    for idx, distance in enumerate(distances):
        if results[distance] is None:
            continue
            
        r = results[distance]
        
        # Get the valid mask for this distance
        valid_mask = r['valid_mask']
        
        # Get valid grid points and projections using the stored valid mask
        valid_grid_flir = grid_points_flir[valid_mask]
        valid_infinity_cctv = r['infinity_pixels']
        valid_finite_cctv = r['finite_pixels']
        
        # Plot 1: FLIR grid points
        ax1 = plt.subplot(n_distances, 4, idx*4 + 1)
        ax1.set_xlim(0, flir_width)
        ax1.set_ylim(flir_height, 0)  # Inverted y-axis for image coordinates
        ax1.scatter(valid_grid_flir[:, 0], valid_grid_flir[:, 1], c='blue', s=10, alpha=0.6)
        ax1.set_title(f'FLIR Grid Points\n({distance}m)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)', fontsize=10)
        ax1.set_ylabel('Y (pixels)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: CCTV projection at infinity
        ax2 = plt.subplot(n_distances, 4, idx*4 + 2)
        ax2.set_xlim(0, cctv_width)
        ax2.set_ylim(cctv_height, 0)
        ax2.scatter(valid_infinity_cctv[:, 0], valid_infinity_cctv[:, 1], 
                   c='green', s=10, alpha=0.6, label='Infinity')
        ax2.set_title(f'CCTV: Infinity Projection\n({distance}m)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)', fontsize=10)
        ax2.set_ylabel('Y (pixels)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
        
        # Plot 3: CCTV projection at actual distance
        ax3 = plt.subplot(n_distances, 4, idx*4 + 3)
        ax3.set_xlim(0, cctv_width)
        ax3.set_ylim(cctv_height, 0)
        ax3.scatter(valid_finite_cctv[:, 0], valid_finite_cctv[:, 1], 
                   c='red', s=10, alpha=0.6, label=f'Actual {distance}m')
        ax3.set_title(f'CCTV: Actual Distance Projection\n({distance}m)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (pixels)', fontsize=10)
        ax3.set_ylabel('Y (pixels)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_aspect('equal')
        
        # Plot 4: Error vectors (showing the delta/difference)
        ax4 = plt.subplot(n_distances, 4, idx*4 + 4)
        ax4.set_xlim(0, cctv_width)
        ax4.set_ylim(cctv_height, 0)
        
        # Draw error vectors from infinity to actual
        pixel_deltas = r['pixel_deltas']
        errors = r['errors']
        
        # Scale vectors for visibility (multiply by a factor)
        scale_factor = 5.0  # Make vectors more visible
        scaled_deltas = pixel_deltas * scale_factor
        
        # Plot infinity points
        ax4.scatter(valid_infinity_cctv[:, 0], valid_infinity_cctv[:, 1], 
                   c='green', s=5, alpha=0.3, label='Infinity')
        
        # Plot error vectors
        for i in range(len(valid_infinity_cctv)):
            if errors[i] > 0:
                start = valid_infinity_cctv[i]
                end = valid_infinity_cctv[i] + scaled_deltas[i]
                # Color by error magnitude
                error_norm = errors[i]
                color_intensity = min(error_norm / np.max(errors), 1.0) if np.max(errors) > 0 else 0
                ax4.arrow(start[0], start[1], scaled_deltas[i, 0], scaled_deltas[i, 1],
                         head_width=10, head_length=10, fc='red', ec='red', 
                         alpha=0.6, linewidth=0.5, length_includes_head=True)
        
        # Color code points by error magnitude
        scatter = ax4.scatter(valid_infinity_cctv[:, 0], valid_infinity_cctv[:, 1], 
                             c=errors, s=30, cmap='hot', alpha=0.8, 
                             vmin=0, vmax=np.max(errors))
        ax4.set_title(f'Error Vectors (Delta)\nMean: {r["mean_error"]:.3f}px', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (pixels)', fontsize=10)
        ax4.set_ylabel('Y (pixels)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        plt.colorbar(scatter, ax=ax4, label='Error (pixels)')
        ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved debug plots to: {output_path}")
    
    return fig

def plot_error_delta_line(results, output_path='infinity_error_delta.png'):
    """
    Create a standalone line plot showing max error delta vs distance.
    
    Args:
        results: Dict with error statistics for each test distance
        output_path: Output file path for the plot
    """
    distances = sorted([d for d in results.keys() if results[d] is not None])
    
    if len(distances) == 0:
        print("No valid results to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract max errors
    max_errors = [results[d]['max_error'] for d in distances]
    
    # Plot line with markers
    ax.plot(distances, max_errors, 'b-o', linewidth=2, markersize=6, 
            label='Max Error (Delta)', markerfacecolor='blue', markeredgecolor='darkblue')
    
    ax.set_xlabel('Distance (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Max Pixel Error (Delta)', fontsize=14, fontweight='bold')
    ax.set_title('Error Delta: Infinity vs Actual Distance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Set x-axis to show all distance increments
    ax.set_xticks(distances[::2])  # Show every other tick to avoid crowding
    ax.tick_params(axis='both', labelsize=11)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved error delta line plot to: {output_path}")
    
    return fig

def plot_results(results, grid_points_flir, output_path='infinity_error_analysis.png'):
    """
    Plot error analysis results for 4m and 100m distances.
    """
    distances = sorted([d for d in results.keys() if results[d] is not None])
    
    if len(distances) == 0:
        print("No valid results to plot")
        return None
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Error comparison bar chart
    ax1 = plt.subplot(2, 3, 1)
    mean_errors = [results[d]['mean_error'] for d in distances]
    max_errors = [results[d]['max_error'] for d in distances]
    
    x_pos = np.arange(len(distances))
    width = 0.35
    ax1.bar(x_pos - width/2, mean_errors, width, label='Mean Error', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, max_errors, width, label='Max Error', color='crimson', alpha=0.8)
    ax1.set_xlabel('Distance (meters)', fontsize=12)
    ax1.set_ylabel('Pixel Error (Delta)', fontsize=12)
    ax1.set_title('Error Delta: Infinity vs Actual Distance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{d}m' for d in distances])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Error distribution comparison
    ax2 = plt.subplot(2, 3, 2)
    colors = ['blue', 'red']
    for i, distance in enumerate(distances):
        errors = results[distance]['errors']
        ax2.hist(errors, bins=50, alpha=0.6, label=f'{distance}m', 
                color=colors[i % len(colors)], density=True)
    ax2.set_xlabel('Pixel Error (Delta)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3-4: Error maps for each distance
    for plot_idx, distance in enumerate(distances[:2], start=3):
        if results[distance] is not None:
            ax = plt.subplot(2, 3, plot_idx)
            
            # Create error map
            errors = results[distance]['errors']
            grid_size = int(np.sqrt(len(errors)))
            error_map = errors.reshape(grid_size, grid_size)
            
            im = ax.imshow(error_map, cmap='hot', interpolation='nearest', aspect='auto')
            ax.set_title(f'Error Map at {distance}m\n(Mean: {results[distance]["mean_error"]:.3f}px)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Grid X', fontsize=10)
            ax.set_ylabel('Grid Y', fontsize=10)
            plt.colorbar(im, ax=ax, label='Pixel Error (Delta)')
    
    # Plot 5: Error statistics table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Create statistics table
    table_data = []
    table_data.append(['Distance (m)', 'Mean (px)', 'Median (px)', 'Max (px)', 'Std (px)'])
    for distance in distances:
        r = results[distance]
        table_data.append([
            f'{distance}',
            f'{r["mean_error"]:.3f}',
            f'{r["median_error"]:.3f}',
            f'{r["max_error"]:.3f}',
            f'{r["std_error"]:.3f}'
        ])
    
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Error Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 6: Error reduction comparison
    ax6 = plt.subplot(2, 3, 6)
    if len(distances) >= 2:
        close_dist = min(distances)
        far_dist = max(distances)
        close_error = results[close_dist]['mean_error']
        far_error = results[far_dist]['mean_error']
        
        reduction = ((close_error - far_error) / close_error * 100) if close_error > 0 else 0
        
        categories = [f'Close ({close_dist}m)', f'Far ({far_dist}m)']
        errors = [close_error, far_error]
        colors_bar = ['#ff6b6b', '#51cf66']
        
        bars = ax6.bar(categories, errors, color=colors_bar, alpha=0.8)
        ax6.set_ylabel('Mean Pixel Error (Delta)', fontsize=12)
        ax6.set_title(f'Error Reduction: {reduction:.1f}%', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.3f}px',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax6.axis('off')
        ax6.text(0.5, 0.5, 'Need at least 2 distances\nfor comparison', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved results plot to: {output_path}")
    
    return fig

def main():
    # Path to sensor calibration file
    calib_path = os.path.join(os.path.dirname(__file__), '..', 'sensor_calibration.txt')
    
    print("Parsing sensor calibration file...")
    cameras, extrinsics, image_dimensions = parse_sensor_calibration(calib_path)
    
    # Get FLIR 8.9MP and CCTV cameras
    flir_name = 'FLIR 8.9MP'
    cctv_name = 'CCTV'
    
    if flir_name not in cameras:
        print(f"Error: {flir_name} not found in calibration file")
        print(f"Available cameras: {list(cameras.keys())}")
        return
    
    if cctv_name not in cameras:
        print(f"Error: {cctv_name} not found in calibration file")
        print(f"Available cameras: {list(cameras.keys())}")
        return
    
    flir_camera = cameras[flir_name]
    cctv_camera = cameras[cctv_name]
    flir_extr = extrinsics[flir_name]
    cctv_extr = extrinsics[cctv_name]
    
    print(f"\nLoaded cameras:")
    print(f"  {flir_name}: {flir_extr['R']}")
    print(f"  {cctv_name}: {cctv_extr['R']}")
    print(f"  FLIR translation: [{flir_extr['X']:.3f}, {flir_extr['Y']:.3f}, {flir_extr['Z']:.3f}]")
    print(f"  CCTV translation: [{cctv_extr['X']:.3f}, {cctv_extr['Y']:.3f}, {cctv_extr['Z']:.3f}]")
    
    # Get image dimensions from calibration or use defaults from sensor_calibration.txt
    # Dimensions are determined by focal lengths and principal points in the calibration
    if flir_name in image_dimensions:
        flir_width, flir_height = image_dimensions[flir_name]
    else:
        # Use defaults from sensor_calibration.txt
        flir_width, flir_height = 4096, 2160
    
    if cctv_name in image_dimensions:
        cctv_width, cctv_height = image_dimensions[cctv_name]
    else:
        # Use defaults from sensor_calibration.txt
        cctv_width, cctv_height = 3840, 2160
    
    print(f"\nImage dimensions:")
    print(f"  {flir_name}: {flir_width}x{flir_height}")
    print(f"  {cctv_name}: {cctv_width}x{cctv_height}")
    
    # Create 20x20 grid and calculate errors at close (4m) and far (100m) distances
    print("\n" + "="*60)
    print("Calculating infinity projection errors...")
    print("Comparing infinity assumption vs actual distances (4m close, 100m far)")
    print("="*60)
    
    # Use distances from 5m to 100m in 5m increments for smoother interpolation curve
    test_distances_detailed = list(range(5, 105, 5))  # [5, 10, 15, ..., 100]
    
    results, grid_points_flir, rays_flir, infinity_pixels = calculate_infinity_error(
        flir_camera, cctv_camera, flir_extr, cctv_extr,
        flir_width, flir_height, grid_size=20,
        test_distances=test_distances_detailed  # 5m to 100m in 5m increments
    )
    
    # Also calculate for the key distances (4m, 20m, 100m) for debug plots
    results_debug, _, _, _ = calculate_infinity_error(
        flir_camera, cctv_camera, flir_extr, cctv_extr,
        flir_width, flir_height, grid_size=20,
        test_distances=[4, 20, 100]  # For debug plots
    )
    
    # Plot results
    output_dir = os.path.dirname(__file__)
    
    # Generate standalone error delta line plot (max error)
    delta_output_path = os.path.join(output_dir, 'infinity_error_delta.png')
    print("\n" + "="*60)
    print("Generating error delta line plot (max error)...")
    print("="*60)
    plot_error_delta_line(results, delta_output_path)
    
    # Generate debug plots (using key distances)
    debug_output_path = os.path.join(output_dir, 'infinity_error_debug.png')
    print("\n" + "="*60)
    print("Generating debug plots...")
    print("="*60)
    plot_debug_results(results_debug, grid_points_flir, infinity_pixels, 
                       flir_width, flir_height, cctv_width, cctv_height, 
                       debug_output_path)
    
    # Generate summary plots (using key distances for summary)
    summary_output_path = os.path.join(output_dir, 'infinity_error_analysis.png')
    print("\n" + "="*60)
    print("Generating summary plots...")
    print("="*60)
    plot_results(results_debug, grid_points_flir, summary_output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Error Delta: Infinity Assumption vs Actual Distance")
    print("="*60)
    distances = sorted([d for d in results.keys() if results[d] is not None])
    for distance in distances:
        r = results[distance]
        print(f"\nDistance: {distance}m")
        print(f"  Mean error (delta): {r['mean_error']:.3f} pixels")
        print(f"  Median error (delta): {r['median_error']:.3f} pixels")
        print(f"  Max error (delta): {r['max_error']:.3f} pixels")
        print(f"  Std error (delta): {r['std_error']:.3f} pixels")
    
    # Compare distances
    if 4 in results and results[4] is not None and 100 in results and results[100] is not None:
        print("\n" + "="*60)
        print("COMPARISON: Close (4m) vs Medium (20m) vs Far (100m)")
        print("="*60)
        if 4 in results and results[4] is not None:
            print(f"4m mean error:  {results[4]['mean_error']:.3f} pixels")
        if 20 in results and results[20] is not None:
            print(f"20m mean error: {results[20]['mean_error']:.3f} pixels")
        if 100 in results and results[100] is not None:
            print(f"100m mean error: {results[100]['mean_error']:.3f} pixels")
        
        if 4 in results and results[4] is not None and 100 in results and results[100] is not None:
            error_reduction = ((results[4]['mean_error'] - results[100]['mean_error']) / results[4]['mean_error'] * 100)
            print(f"\nError reduction from 4m to 100m: {error_reduction:.1f}%")

if __name__ == '__main__':
    main()

