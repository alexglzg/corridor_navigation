"""
Line-pairing based rectangle generation for corridor extraction.

This module provides an alternative to snap-point based rectangle generation.
Instead of traversing snap points, it pairs parallel lines with opposing normals
to form corridor rectangles.
"""

import math
from typing import List, Tuple, Set
from itertools import combinations


def lines_are_parallel(line1, line2, angle_tol_deg: float = 5.0) -> bool:
    """
    Check if two lines are parallel within angle tolerance.
    
    Lines are parallel if their direction vectors point the same way
    (or opposite, since we care about the axis, not direction along it).
    """
    # Get direction vectors from endpoints
    d1 = (line1.p2[0] - line1.p1[0], line1.p2[1] - line1.p1[1])
    d2 = (line2.p2[0] - line2.p1[0], line2.p2[1] - line2.p1[1])
    
    # Normalize
    len1 = math.hypot(*d1)
    len2 = math.hypot(*d2)
    if len1 < 1e-6 or len2 < 1e-6:
        return False
    
    d1 = (d1[0] / len1, d1[1] / len1)
    d2 = (d2[0] / len2, d2[1] / len2)
    
    # Check if parallel (dot product close to 1 or -1)
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    angle_diff = math.acos(max(-1, min(1, abs(dot))))
    
    return math.degrees(angle_diff) <= angle_tol_deg


def normals_face_each_other(line1, line2) -> bool:
    """
    Check if two lines' normals point toward each other.
    
    This verifies that:
    - line1's normal points toward line2
    - line2's normal points toward line1
    
    This is NOT the same as checking if normals are anti-parallel.
    For obstacle edges, normals point outward (away from each other).
    For room walls, normals point inward (toward each other).
    """
    # Compute midpoints
    mid1 = ((line1.p1[0] + line1.p2[0]) / 2, (line1.p1[1] + line1.p2[1]) / 2)
    mid2 = ((line2.p1[0] + line2.p2[0]) / 2, (line2.p1[1] + line2.p2[1]) / 2)
    
    # Vector from line1 to line2
    v12 = (mid2[0] - mid1[0], mid2[1] - mid1[1])
    
    n1 = line1.normal
    n2 = line2.normal
    
    # line1's normal should point toward line2 (positive dot with v12)
    dot1 = n1[0] * v12[0] + n1[1] * v12[1]
    
    # line2's normal should point toward line1 (negative dot with v12, or positive with v21)
    dot2 = n2[0] * (-v12[0]) + n2[1] * (-v12[1])
    
    return dot1 > 0 and dot2 > 0


def corridor_is_free_space(rect, grid_image, threshold: int = 250) -> bool:
    """
    Check if a candidate corridor rectangle represents free space.
    
    Samples points inside the rectangle and verifies they're not obstacles
    (white pixels in the grid image = free space).
    
    Parameters:
    - rect: Rectangle object with center and half_extents
    - grid_image: Binary occupancy grid (white=free, black=obstacle)
    - threshold: Pixel value threshold for free space
    
    Returns:
    - True if the corridor is mostly free space
    """
    import numpy as np
    
    height, width = grid_image.shape[:2]
    
    # Sample points inside the rectangle
    cx, cy = rect.center
    hx, hy = rect.half_extents
    
    # Sample grid (5x5 points inside the rectangle)
    num_samples = 5
    obstacle_count = 0
    total_samples = 0
    
    for i in range(num_samples):
        for j in range(num_samples):
            # Interpolate position within rectangle
            fx = (i / (num_samples - 1)) * 2 - 1  # -1 to 1
            fy = (j / (num_samples - 1)) * 2 - 1  # -1 to 1
            
            # Apply some margin to avoid sampling exactly on the boundary
            fx *= 0.8
            fy *= 0.8
            
            px = int(cx + fx * hx)
            py = int(cy + fy * hy)
            
            # Check bounds
            if 0 <= px < width and 0 <= py < height:
                total_samples += 1
                # Check if pixel is obstacle (dark)
                if grid_image[py, px] < threshold:
                    obstacle_count += 1
    
    if total_samples == 0:
        return False
    
    # Allow some tolerance (e.g., up to 20% of samples can be obstacles due to wall thickness)
    obstacle_ratio = obstacle_count / total_samples
    return obstacle_ratio < 0.2


def line_blocks_pair(blocker, line1, line2, angle_tol_deg: float = 5.0) -> bool:
    """
    Check if 'blocker' line lies between line1 and line2 and would block a corridor.
    
    A blocker must be:
    - Parallel to line1 and line2
    - Have a normal facing one of them (i.e., it's a boundary in between)
    - Geometrically between line1 and line2 along the perpendicular axis
    - Overlapping with both line1 and line2 along the parallel axis
    """
    # Must be parallel to the pair
    if not lines_are_parallel(blocker, line1, angle_tol_deg):
        return False
    
    # Get midpoints
    mid1 = ((line1.p1[0] + line1.p2[0]) / 2, (line1.p1[1] + line1.p2[1]) / 2)
    mid2 = ((line2.p1[0] + line2.p2[0]) / 2, (line2.p1[1] + line2.p2[1]) / 2)
    mid_b = ((blocker.p1[0] + blocker.p2[0]) / 2, (blocker.p1[1] + blocker.p2[1]) / 2)
    
    # Project onto normal direction to check if blocker is between line1 and line2
    n1 = line1.normal
    
    # Project midpoints onto normal
    proj1 = mid1[0] * n1[0] + mid1[1] * n1[1]
    proj2 = mid2[0] * n1[0] + mid2[1] * n1[1]
    proj_b = mid_b[0] * n1[0] + mid_b[1] * n1[1]
    
    # Blocker must be strictly between line1 and line2
    min_proj, max_proj = min(proj1, proj2), max(proj1, proj2)
    margin = 1.0  # Small margin to avoid floating point issues
    if not (min_proj + margin < proj_b < max_proj - margin):
        return False
    
    # Check that blocker overlaps with both lines along the parallel axis
    # (otherwise it's not actually blocking the corridor)
    overlap1 = compute_overlap_extent(line1, blocker)
    overlap2 = compute_overlap_extent(line2, blocker)
    
    if overlap1 is None or overlap2 is None:
        return False
    
    # Blocker's normal should face toward at least one of the lines
    # (meaning it's a boundary, not just a random parallel line)
    nb = blocker.normal
    v1b = (mid1[0] - mid_b[0], mid1[1] - mid_b[1])
    v2b = (mid2[0] - mid_b[0], mid2[1] - mid_b[1])
    
    faces_line1 = (nb[0] * v1b[0] + nb[1] * v1b[1]) > 0
    faces_line2 = (nb[0] * v2b[0] + nb[1] * v2b[1]) > 0
    
    # A blocker should face at least one of the lines
    return faces_line1 or faces_line2


def compute_perpendicular_distance(line1, line2) -> float:
    """
    Compute the perpendicular distance between two parallel lines.
    
    Uses the normal of line1 to project the distance.
    """
    # Use midpoint of line1 and project onto line2's plane
    mid1 = ((line1.p1[0] + line1.p2[0]) / 2, (line1.p1[1] + line1.p2[1]) / 2)
    mid2 = ((line2.p1[0] + line2.p2[0]) / 2, (line2.p1[1] + line2.p2[1]) / 2)
    
    # Vector from mid1 to mid2
    diff = (mid2[0] - mid1[0], mid2[1] - mid1[1])
    
    # Project onto normal direction
    n1 = line1.normal
    dist = abs(diff[0] * n1[0] + diff[1] * n1[1])
    
    return dist


def compute_overlap_extent(line1, line2) -> Tuple[float, float, float, float] | None:
    """
    Compute the overlapping extent of two parallel lines along their shared axis.
    
    Returns the bounding box (x_min, y_min, x_max, y_max) of the corridor,
    or None if lines don't overlap along their axis.
    """
    # Get direction vector (use line1's direction)
    d = (line1.p2[0] - line1.p1[0], line1.p2[1] - line1.p1[1])
    length = math.hypot(*d)
    if length < 1e-6:
        return None
    d = (d[0] / length, d[1] / length)
    
    # Project all endpoints onto the axis direction
    # Using line1.p1 as origin for projection
    origin = line1.p1
    
    def project(p):
        return (p[0] - origin[0]) * d[0] + (p[1] - origin[1]) * d[1]
    
    # Get projections
    t1_a = project(line1.p1)
    t1_b = project(line1.p2)
    t2_a = project(line2.p1)
    t2_b = project(line2.p2)
    
    # Get ranges
    t1_min, t1_max = min(t1_a, t1_b), max(t1_a, t1_b)
    t2_min, t2_max = min(t2_a, t2_b), max(t2_a, t2_b)
    
    # Compute overlap
    overlap_min = max(t1_min, t2_min)
    overlap_max = min(t1_max, t2_max)
    
    if overlap_max <= overlap_min:
        return None  # No overlap
    
    return overlap_min, overlap_max, d, origin


def compute_corridor_corners(line1, line2, overlap_info) -> List[Tuple[float, float]] | None:
    """
    Compute the four corners of the corridor rectangle.
    
    The corridor is bounded by:
    - The two lines (perpendicular extent)
    - The overlap region (parallel extent)
    """
    if overlap_info is None:
        return None
    
    overlap_min, overlap_max, d, origin = overlap_info
    
    # Get normal direction (perpendicular to d)
    n = (-d[1], d[0])
    
    # Compute the two "sides" of the corridor along the normal
    mid1 = ((line1.p1[0] + line1.p2[0]) / 2, (line1.p1[1] + line1.p2[1]) / 2)
    mid2 = ((line2.p1[0] + line2.p2[0]) / 2, (line2.p1[1] + line2.p2[1]) / 2)
    
    # Project midpoints onto normal to determine which line is on which side
    n1_proj = (mid1[0] - origin[0]) * n[0] + (mid1[1] - origin[1]) * n[1]
    n2_proj = (mid2[0] - origin[0]) * n[0] + (mid2[1] - origin[1]) * n[1]
    
    # Four corners of the corridor
    # Corner = origin + t * d + s * n
    # where t is the parallel coordinate and s is the perpendicular coordinate
    
    corners = [
        (origin[0] + overlap_min * d[0] + n1_proj * n[0],
         origin[1] + overlap_min * d[1] + n1_proj * n[1]),
        (origin[0] + overlap_max * d[0] + n1_proj * n[0],
         origin[1] + overlap_max * d[1] + n1_proj * n[1]),
        (origin[0] + overlap_max * d[0] + n2_proj * n[0],
         origin[1] + overlap_max * d[1] + n2_proj * n[1]),
        (origin[0] + overlap_min * d[0] + n2_proj * n[0],
         origin[1] + overlap_min * d[1] + n2_proj * n[1]),
    ]
    
    return corners


def generate_rects_by_line_pairing(linemap, 
                                    max_corridor_width: float = None,
                                    min_corridor_width: float = None,
                                    angle_tolerance: float = 5.0) -> None:
    """
    Generate rectangles by pairing parallel lines with opposing normals.
    
    This replaces the snap-point based rectangle generation for experimental
    corridor extraction that handles:
    - Doors at corners (where half-snap extension fails)
    - Structured obstacles like warehouse aisles
    
    Parameters:
    - linemap: LineMap instance with lines already detected and straightened
    - max_corridor_width: Maximum distance between paired lines (pixels)
    - min_corridor_width: Minimum distance between paired lines (pixels)
    - angle_tolerance: Tolerance for considering lines parallel (degrees)
    """
    from rectangle import Rectangle
    
    if not linemap.lines:
        raise RuntimeError("No lines available. Run detect_lines() and straighten_lines() first.")
    
    lines = list(linemap.lines)
    map_bounds = (linemap.grid_image.shape[1], linemap.grid_image.shape[0])
    
    # Default corridor width constraints
    if min_corridor_width is None:
        min_corridor_width = linemap.min_rect_size * 2
    if max_corridor_width is None:
        max_corridor_width = max(map_bounds)  # Full map dimension
    
    if linemap.debug:
        print(f"Line pairing: {len(lines)} lines, "
              f"width range [{min_corridor_width:.1f}, {max_corridor_width:.1f}]")
    
    candidates = []
    
    # Stats for debugging
    stats = {
        'pairs_checked': 0,
        'not_parallel': 0,
        'normals_not_facing': 0,
        'width_out_of_range': 0,
        'no_overlap': 0,
        'blocked': 0,
        'not_free_space': 0,
        'rectangle_failed': 0,
    }
    
    # Check all pairs of lines
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if j <= i:
                continue
            
            stats['pairs_checked'] += 1
            
            # Check if parallel
            if not lines_are_parallel(line1, line2, angle_tolerance):
                stats['not_parallel'] += 1
                continue
            
            # Check if normals face each other (not just anti-parallel)
            if not normals_face_each_other(line1, line2):
                stats['normals_not_facing'] += 1
                continue
            
            # Check corridor width
            width = compute_perpendicular_distance(line1, line2)
            if width < min_corridor_width or width > max_corridor_width:
                stats['width_out_of_range'] += 1
                continue
            
            # Check if any other line blocks this pair
            blocked = False
            for k, blocker in enumerate(lines):
                if k == i or k == j:
                    continue
                if line_blocks_pair(blocker, line1, line2, angle_tolerance):
                    blocked = True
                    break
            
            if blocked:
                stats['blocked'] += 1
                continue
            
            # Compute overlap extent
            overlap_info = compute_overlap_extent(line1, line2)
            if overlap_info is None:
                stats['no_overlap'] += 1
                continue
            
            # Compute corridor corners
            corners = compute_corridor_corners(line1, line2, overlap_info)
            if corners is None:
                stats['no_overlap'] += 1
                continue
            
            # Try to create rectangle
            try:
                rect = Rectangle(corners[0], corners[1], corners[2], map_bounds)
                
                # Validate that corridor represents free space (not crossing walls)
                if not corridor_is_free_space(rect, linemap.grid_image):
                    stats['not_free_space'] += 1
                    if linemap.debug:
                        print(f"  Rejected (not free space): L{i} + L{j}")
                    continue
                
                candidates.append((rect, line1, line2, width))
                if linemap.debug:
                    print(f"  Paired: L{i}({line1.p1}->{line1.p2}) + L{j}({line2.p1}->{line2.p2}), width={width:.1f}")
            except ValueError as e:
                stats['rectangle_failed'] += 1
                if linemap.debug:
                    print(f"  Rectangle failed: {e}")
    
    if linemap.debug:
        print(f"Pairing stats: {stats}")
        print(f"Found {len(candidates)} candidate corridors")
    
    # Add valid rectangles
    for rect, line1, line2, width in candidates:
        if (rect.half_extents[0] >= linemap.min_rect_size and 
            rect.half_extents[1] >= linemap.min_rect_size):
            linemap.rectangles.add_node(rect)
    
    if linemap.debug:
        print(f"Added {len(linemap.rectangles.nodes())} rectangles")