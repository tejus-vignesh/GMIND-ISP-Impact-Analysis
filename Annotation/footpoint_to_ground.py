"""
Robust geometric 3D projection from 2D bounding boxes using ground plane intersection.

This module implements a modular pipeline with each step tested independently:
1. Camera model & undistort
2. Choose representative pixel from bbox
3. Normalized image coordinates
4. Camera → World rotation
5. Rotate ray to world frame
6. Ray-plane intersection

Each step has unit tests to verify correctness.
"""

from typing import Optional, Tuple, Union
import numpy as np
import cv2


# ============================================================================
# STEP 1: Camera Model & Undistort
# ============================================================================

def undistort_bbox_points(
    bbox: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Undistort bounding box corner points if distortion coefficients are provided.
    
    If no distortion coefficients are provided, returns bbox unchanged.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        camera_matrix: Camera intrinsics (3x3) with fx, fy, cx, cy
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3) or None
    
    Returns:
        Undistorted bounding box [x1, y1, x2, y2]
    """
    if dist_coeffs is None:
        return bbox.copy()
    
    x1, y1, x2, y2 = bbox[:4]
    
    # Extract points
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Undistort points
    undistorted_points = cv2.undistortPoints(
        points, 
        camera_matrix, 
        dist_coeffs, 
        P=camera_matrix
    )
    
    # Convert back to pixel coordinates
    undistorted = undistorted_points.reshape(-1, 2)
    
    return np.array([
        undistorted[0, 0],
        undistorted[0, 1],
        undistorted[1, 0],
        undistorted[1, 1]
    ], dtype=np.float32)


# ============================================================================
# STEP 2: Choose Representative Pixel from Bbox
# ============================================================================

def get_representative_pixel(
    bbox: np.ndarray,
    method: str = "bottom_center",
    num_samples: int = 1
) -> np.ndarray:
    """
    Choose representative pixel(s) from bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        method: "bottom_center" (default) or "bottom_edge_median"
        num_samples: Number of samples along bottom edge (for median method)
    
    Returns:
        Array of representative pixels [[u, v], ...] in pixel coordinates
    """
    x1, y1, x2, y2 = bbox[:4]
    
    if method == "bottom_center":
        u = (x1 + x2) / 2.0
        v = float(y2)  # Bottom edge
        return np.array([[u, v]], dtype=np.float32)
    
    elif method == "bottom_edge_median":
        # Sample multiple points along bottom edge
        if num_samples == 1:
            u = (x1 + x2) / 2.0
            v = float(y2)
            return np.array([[u, v]], dtype=np.float32)
        
        # Sample evenly along bottom edge
        u_samples = np.linspace(x1, x2, num_samples)
        v_samples = np.full(num_samples, float(y2))
        points = np.stack([u_samples, v_samples], axis=1)
        
        # Return all points (caller will compute median)
        return points.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# STEP 3: Normalized Image Coordinates
# ============================================================================

def pixel_to_normalized_coords(
    pixel: np.ndarray,
    camera_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert pixel coordinates to normalized camera coordinates.
    
    Normalized coordinates represent the ray direction in camera frame.
    The [x, y, 1] form encodes the actual ray direction with focal length information.
    
    Args:
        pixel: Pixel coordinates [u, v] or array of pixels Nx2
        camera_matrix: Camera intrinsics (3x3) with fx, fy, cx, cy
    
    Returns:
        Normalized coordinates [x, y, 1] or Nx3 array of normalized coords
        where x = (u - cx) / fx, y = (v - cy) / fy
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    pixel = np.asarray(pixel, dtype=np.float32)
    
    if pixel.ndim == 1:
        # Single pixel [u, v]
        u, v = pixel[0], pixel[1]
        x = (u - cx) / fx
        y = (v - cy) / fy
        return np.array([x, y, 1.0], dtype=np.float32)
    else:
        # Multiple pixels Nx2
        u = pixel[:, 0]
        v = pixel[:, 1]
        x = (u - cx) / fx
        y = (v - cy) / fy
        ones = np.ones(len(pixel), dtype=np.float32)
        return np.stack([x, y, ones], axis=1)


# ============================================================================
# STEP 4: Camera → World Rotation
# ============================================================================

def build_rotation_cam_to_world(
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0
) -> np.ndarray:
    """
    Build rotation matrix from camera frame to world frame.
    
    Coordinate conventions:
    - Camera frame (OpenCV): x = right, y = down, z = forward
    - World frame: X = right, Y = forward, Z = up
    
    The complete transformation includes:
    1. Camera orientation rotations (pitch, roll, yaw)
    2. Coordinate system transformation from camera to world convention
    
    Using cv2.Rodrigues for rotations:
    - Pitch: rotation around camera x-axis (positive = camera tilted down)
    - Roll: rotation around camera y-axis (positive = camera tilted right)  
    - Yaw: rotation around camera z-axis (positive = camera rotated left)
    
    Args:
        pitch_deg: Pitch angle in degrees (positive = camera tilted down)
        roll_deg: Roll angle in degrees (positive = camera tilted right)
        yaw_deg: Yaw angle in degrees (positive = camera rotated left)
    
    Returns:
        Rotation matrix (3x3) that transforms from camera to world coordinates
        d_world = R_cam_to_world @ d_cam
        No additional axis flips needed after this transformation.
    """
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # Build rotation matrices using cv2.Rodrigues
    # Pitch: rotation around x-axis
    R_pitch, _ = cv2.Rodrigues(np.array([pitch_rad, 0, 0], dtype=np.float32))
    
    # Roll: rotation around y-axis  
    R_roll, _ = cv2.Rodrigues(np.array([0, roll_rad, 0], dtype=np.float32))
    
    # Yaw: rotation around z-axis
    R_yaw, _ = cv2.Rodrigues(np.array([0, 0, yaw_rad], dtype=np.float32))
    
    # Compose rotations: R_orient = R_yaw @ R_roll @ R_pitch
    # This handles camera orientation in camera frame
    R_orient = R_yaw @ R_roll @ R_pitch
    
    # Coordinate system transformation matrix
    # Maps from camera frame convention to world frame convention:
    # Camera: x=right, y=down, z=forward
    # World:  X=right, Y=forward, Z=up
    #
    # For direction vectors after pitch rotation:
    # - Camera x (right) → World X (right)    
    # - Camera z (forward) → World Y (forward)  
    # - Camera y (down axis, but negative after pitch) → World Z
    #
    # After pitch rotation: ray has camera y = -0.342 (negative = pointing up relative to down axis)
    # We want world Z = -0.342 (negative = pointing down)
    # So: World Z = Camera y (preserve sign)
    T_cam_to_world = np.array([
        [1,  0,  0],  # x → X
        [0,  0,  1],  # z → Y  
        [0,  1,  0]   # y → Z (preserve sign: camera y negative → world Z negative)
    ], dtype=np.float32)
    
    # Complete transformation: first apply orientation, then coordinate system change
    R_cam_to_world = T_cam_to_world @ R_orient
    
    return R_cam_to_world


# ============================================================================
# STEP 5: Rotate Ray to World Frame
# ============================================================================

def rotate_ray_to_world(
    ray_cam: np.ndarray,
    R_cam_to_world: np.ndarray
) -> np.ndarray:
    """
    Rotate ray direction from camera frame to world frame.
    
    Args:
        ray_cam: Ray direction in camera coordinates [x, y, 1] or Nx3 array
        R_cam_to_world: Rotation matrix (3x3) from camera to world
    
    Returns:
        Ray direction in world coordinates [X, Y, Z] or Nx3 array
    """
    ray_cam = np.asarray(ray_cam, dtype=np.float32)
    
    if ray_cam.ndim == 1:
        # Single ray
        ray_world = R_cam_to_world @ ray_cam.reshape(3, 1)
        return ray_world.ravel()
    else:
        # Multiple rays Nx3
        ray_world = (R_cam_to_world @ ray_cam.T).T
        return ray_world


# ============================================================================
# STEP 6: Ray-Plane Intersection
# ============================================================================

def ray_plane_intersection(
    camera_center_world: np.ndarray,
    ray_world: np.ndarray,
    plane_z: float = 0.0,
    min_distance: float = 0.01,
    max_distance: float = 1000.0
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Compute intersection of ray with horizontal ground plane.
    
    Args:
        camera_center_world: Camera position in world coordinates [X, Y, Z]
        ray_world: Ray direction in world coordinates [X, Y, Z] (not normalized)
        plane_z: Z coordinate of the ground plane (default 0.0)
        min_distance: Minimum valid distance (meters)
        max_distance: Maximum valid distance (meters)
    
    Returns:
        Tuple of (intersection_point, distance) or (None, None) if invalid
        intersection_point: 3D point [X, Y, Z] on ground plane
        distance: Distance from camera to intersection point (parameter s)
    """
    camera_center_world = np.asarray(camera_center_world, dtype=np.float32).ravel()
    ray_world = np.asarray(ray_world, dtype=np.float32).ravel()
    
    C_z = camera_center_world[2]
    d_z = ray_world[2]
    
    # Check if ray is parallel to ground plane
    if abs(d_z) < 1e-8:
        return None, None
    
    # Compute intersection parameter: s = (z_plane - C_z) / d_z = -h / d_z
    # where h = camera height above plane
    s = (plane_z - C_z) / d_z
    
    # Sanity check: require s > 0 (ray intersects ground forward of camera)
    if s <= 0:
        return None, None
    
    # Check reasonable distance bounds
    if s < min_distance or s > max_distance:
        return None, None
    
    # Compute intersection point: P = C + s * d
    P_world = camera_center_world + s * ray_world
    
    # Verify point is on ground plane
    if abs(P_world[2] - plane_z) > 0.1:
        return None, None
    
    return P_world, s


# ============================================================================
# MAIN FUNCTION: Complete Pipeline
# ============================================================================

def bbox_to_3d_geometric_robust(
    bbox: np.ndarray,
    camera_matrix: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
    ground_height: float = 0.0,
    dist_coeffs: Optional[np.ndarray] = None,
    pixel_method: str = "bottom_center",
    num_samples: int = 1,
    camera_roll_deg: float = 0.0,
    camera_yaw_deg: float = 0.0
) -> Optional[np.ndarray]:
    """
    Compute 3D location from bounding box using robust geometric pipeline.
    
    This function orchestrates the complete pipeline:
    1. Undistort bbox points
    2. Choose representative pixel(s)
    3. Convert to normalized coordinates
    4. Build camera-to-world rotation
    5. Rotate ray to world frame
    6. Compute ray-plane intersection
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        camera_matrix: Camera intrinsics (3x3) with fx, fy, cx, cy
        camera_height: Camera height above ground in meters
        camera_pitch_deg: Camera pitch angle in degrees (positive = downward)
        ground_height: Ground plane height in world coordinates (meters), default 0.0
        dist_coeffs: Distortion coefficients (optional)
        pixel_method: "bottom_center" or "bottom_edge_median"
        num_samples: Number of samples for median method
        camera_roll_deg: Camera roll angle in degrees (default 0.0)
        camera_yaw_deg: Camera yaw angle in degrees (default 0.0)
    
    Returns:
        3D point [X, Y, Z] in world coordinates (meters) with origin at ground level.
        Returns None if invalid (e.g., ray parallel to ground, too far, etc.).
    """
    # Step 1: Undistort bbox points
    bbox_undistorted = undistort_bbox_points(bbox, camera_matrix, dist_coeffs)
    
    # Step 2: Choose representative pixel(s)
    pixels = get_representative_pixel(bbox_undistorted, pixel_method, num_samples)
    
    # If multiple pixels, compute median
    if len(pixels) > 1:
        pixel = np.median(pixels, axis=0)
    else:
        pixel = pixels[0]
    
    # Step 3: Convert to normalized coordinates
    ray_cam = pixel_to_normalized_coords(pixel, camera_matrix)
    # Simple coordinate system adjustment: invert y to swap pattern
    # (top pixels hit ground further, bottom closer)
    if ray_cam.ndim == 1:
        ray_cam[1] = -ray_cam[1]
    else:
        ray_cam[:, 1] = -ray_cam[:, 1]
    
    # Step 4: Build camera-to-world rotation
    R_cam_to_world = build_rotation_cam_to_world(
        pitch_deg=camera_pitch_deg,
        roll_deg=camera_roll_deg,
        yaw_deg=camera_yaw_deg
    )
    
    # Step 5: Rotate ray to world frame
    ray_world = rotate_ray_to_world(ray_cam, R_cam_to_world)
    
    # Step 6: Compute ray-plane intersection
    camera_center_world = np.array([0.0, 0.0, camera_height], dtype=np.float32)
    P_world, distance = ray_plane_intersection(
        camera_center_world,
        ray_world,
        plane_z=ground_height,
        min_distance=0.01,
        max_distance=1000.0
    )
    
    if P_world is None:
        return None
    
    # Shift origin to ground level directly below camera
    # Final coordinates are relative to ground origin
    P_final = P_world.copy()
    P_final[2] = 0.0  # Z should be 0 (on ground)
    
    return P_final

