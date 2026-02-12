"""
Extended snap point logic for handling:
1. Door-at-corner: half snap encountering a full snap
2. Obstacle edges: extending short lines to room walls

These functions are designed to be called from LineMap methods.
"""

import math
from typing import Set, List, Tuple, Optional
from util import (vec_sub, vec_add, vec_scale, vec_normalize, vec_dot, 
                   vec_length_sq, ray_line_intersection, EPS, MIN_INTERSECTION_DIST_SQ)
from snappoint import HalfSnapPoint, FullSnapPoint
from line import Line


def find_full_snap_with_perpendicular_wall(
    origin: Tuple[float, float],
    ray_dir: Tuple[float, float],
    full_snaps: Set,
    current_line: Line,
    max_distance: float,
    proximity_threshold: float = 10.0
) -> Tuple[Optional[FullSnapPoint], Optional[Line], Optional[float]]:
    """
    Check if ray passes near a full snap that has a wall perpendicular to the ray.
    
    This handles the door-at-corner case where:
    - A half snap H casts a ray
    - The ray passes near a full snap F (corner)
    - F has a wall W perpendicular to the ray direction
    - W becomes the "hit line" for H
    
    Returns:
        (full_snap, perpendicular_wall, distance) or (None, None, None)
    """
    best_snap = None
    best_wall = None
    best_dist = float('inf')
    
    for fs in full_snaps:
        if len(fs.lines) != 2:
            continue
            
        # Vector from origin to full snap
        to_fs = (fs.x - origin[0], fs.y - origin[1])
        
        # Project onto ray direction
        t = vec_dot(to_fs, ray_dir)
        if t <= 0 or t > max_distance:
            continue
        
        # Perpendicular distance from ray to full snap
        closest_on_ray = (origin[0] + t * ray_dir[0], origin[1] + t * ray_dir[1])
        perp_dist_sq = (fs.x - closest_on_ray[0])**2 + (fs.y - closest_on_ray[1])**2
        
        if perp_dist_sq > proximity_threshold**2:
            continue
        
        # Check if F has a wall perpendicular to ray direction
        for wall in fs.lines:
            if wall is current_line:
                continue
            
            # Get wall direction
            wall_dir = vec_normalize(vec_sub(wall.p2, wall.p1))
            
            # Check if wall is perpendicular to ray (dot product ~0)
            dot = abs(vec_dot(wall_dir, ray_dir))
            if dot < 0.3:  # Roughly perpendicular (within ~70 degrees)
                # Check if wall's normal faces toward the ray origin
                wall_mid = ((wall.p1[0] + wall.p2[0])/2, (wall.p1[1] + wall.p2[1])/2)
                to_origin = vec_sub(origin, wall_mid)
                if vec_dot(wall.normal, to_origin) > 0:
                    if t < best_dist:
                        best_snap = fs
                        best_wall = wall
                        best_dist = t
    
    if best_snap is not None:
        return best_snap, best_wall, best_dist
    return None, None, None


def extend_half_snap_to_full_snap_wall(
    current_sp: HalfSnapPoint,
    current_line: Line,
    full_snap: FullSnapPoint,
    perpendicular_wall: Line,
    distance: float,
    ray_dir: Tuple[float, float]
) -> None:
    """
    Extend a half snap to connect with a full snap's perpendicular wall.
    
    This creates the door-at-corner corridor connection.
    """
    origin = (current_sp.x, current_sp.y)
    
    # Move the half snap to the intersection point
    ix = origin[0] + ray_dir[0] * distance
    iy = origin[1] + ray_dir[1] * distance
    
    current_sp.x = int(round(ix))
    current_sp.y = int(round(iy))
    current_sp.hit_line = perpendicular_wall
    perpendicular_wall.hit_half_snaps.add(current_sp)
    
    # Update the line's endpoints
    try:
        current_line.extend(*current_line.snap_points)
    except Exception as e:
        raise RuntimeError(f"Failed extending to full snap wall: {e}")


def identify_obstacle_lines(
    lines: Set[Line],
    half_snaps: Set[HalfSnapPoint],
    full_snaps: Set[FullSnapPoint],
    map_bounds: Tuple[int, int],
    margin: float = 20.0
) -> List[Line]:
    """
    Identify lines that are likely obstacle edges (not room walls).
    
    Heuristics:
    - Not touching map boundary
    - Both endpoints are half snaps (hanging) OR
    - Short compared to room walls
    """
    width, height = map_bounds
    obstacle_lines = []
    
    for line in lines:
        # Check if line touches boundary
        touches_boundary = False
        for p in [line.p1, line.p2]:
            if (p[0] <= margin or p[0] >= width - margin or
                p[1] <= margin or p[1] >= height - margin):
                touches_boundary = True
                break
        
        if touches_boundary:
            continue
        
        # Check if both endpoints are half snaps
        sp1, sp2 = line.snap_points
        both_half = (isinstance(sp1, HalfSnapPoint) and isinstance(sp2, HalfSnapPoint))
        
        if both_half:
            obstacle_lines.append(line)
            continue
        
        # Also include lines where at least one endpoint is a half snap
        # and the line is short (likely an obstacle edge)
        line_length = math.sqrt(line.length_sq())
        if line_length < min(width, height) / 4:
            if isinstance(sp1, HalfSnapPoint) or isinstance(sp2, HalfSnapPoint):
                obstacle_lines.append(line)
    
    return obstacle_lines


def extend_obstacle_line_to_walls(
    obstacle_line: Line,
    all_lines: Set[Line],
    linemap,
    min_extension: float = 10.0
) -> List[Line]:
    """
    Extend an obstacle line along its direction until it hits room walls.
    
    This creates "virtual walls" that span from room wall to room wall,
    enabling corridor formation across the full aisle width.
    
    Returns:
        List of new extension lines created
    """
    from .line import Line as LineClass
    
    # Get line direction
    line_dir = vec_normalize(vec_sub(obstacle_line.p2, obstacle_line.p1))
    line_normal = obstacle_line.normal
    
    new_lines = []
    
    # For each endpoint, extend outward until hitting a wall
    for endpoint, extend_dir in [(obstacle_line.p1, (-line_dir[0], -line_dir[1])),
                                  (obstacle_line.p2, line_dir)]:
        
        # Cast ray from endpoint in extend direction
        best_t = float('inf')
        best_wall = None
        
        for wall in all_lines:
            if wall is obstacle_line:
                continue
            if getattr(wall, 'is_extension', False):
                continue
            
            # Wall must be roughly perpendicular to extension direction
            wall_dir = vec_normalize(vec_sub(wall.p2, wall.p1))
            if abs(vec_dot(wall_dir, extend_dir)) > 0.3:
                continue
            
            # Find intersection
            for seg_start, seg_end in wall.orig_segments:
                t = ray_line_intersection(endpoint, extend_dir, seg_start, seg_end)
                if t is not None and t > min_extension and t < best_t:
                    best_t = t
                    best_wall = wall
        
        if best_wall is not None and best_t < float('inf'):
            # Create extension from endpoint to wall intersection
            end_point = (endpoint[0] + extend_dir[0] * best_t,
                        endpoint[1] + extend_dir[1] * best_t)
            
            # Create a new line segment for the extension
            # The extension has the same normal as the original obstacle line
            ext_line = LineClass(endpoint, end_point, line_normal)
            ext_line.is_extension = True
            ext_line.parent_obstacle = obstacle_line
            new_lines.append(ext_line)
    
    return new_lines


def create_extended_aisle_lines(
    linemap,
    debug: bool = False
) -> None:
    """
    Main function to extend obstacle edges for aisle corridor formation.
    
    This finds obstacle lines and extends them to room walls,
    effectively creating the full-width aisle boundaries.
    """
    map_bounds = (linemap.grid_image.shape[1], linemap.grid_image.shape[0])
    
    # Identify obstacle lines
    obstacle_lines = identify_obstacle_lines(
        linemap.lines,
        linemap.half_snaps,
        linemap.full_snaps,
        map_bounds
    )
    
    if debug:
        print(f"Found {len(obstacle_lines)} potential obstacle lines")
    
    # Extend each obstacle line
    all_extensions = []
    for obs_line in obstacle_lines:
        extensions = extend_obstacle_line_to_walls(
            obs_line,
            linemap.lines,
            linemap
        )
        all_extensions.extend(extensions)
        if debug and extensions:
            print(f"  Extended {obs_line.p1}->{obs_line.p2}: {len(extensions)} extensions")
    
    # Add extensions to linemap
    # Note: We don't add them to self.lines directly; instead we create
    # virtual corridor boundaries that can be used for rectangle generation
    linemap.extension_lines = all_extensions
    
    if debug:
        print(f"Created {len(all_extensions)} extension lines for aisle corridors")


def get_extended_lines_for_pairing(linemap) -> List[Line]:
    """
    Get all lines including extensions for corridor generation.
    
    This combines:
    - Original room wall lines
    - Obstacle edge lines  
    - Extension lines (virtual boundaries for aisles)
    """
    all_lines = list(linemap.lines)
    
    # Add extension lines if they exist
    if hasattr(linemap, 'extension_lines'):
        all_lines.extend(linemap.extension_lines)
    
    return all_lines