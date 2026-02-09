import math
from typing import List, Tuple, Union

import cv2

# Configuration guide for Line Segment Detector (LSD) in OpenCV
# LSD is sensitive to noise, resolution, and line styles, so tune carefully.
# More info: https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html (Version 4.11.0 was used for testing)

# General tuning strategy:
# - If thin walls or faint lines are missed:
#     -> raise `scale` (closer to 1.0), lower `quant`, or lower `density_th`
# - If false or imaginary lines appear (e.g. from textures, text):
#     -> raise `sigma_scale`, raise `density_th`, or lower `ang_th`

# - cv2.LSD_REFINE_NONE: no refinement (raw endpoints, possibly jagged)
# - cv2.LSD_REFINE_STD: standard refinement (trims ends for cleaner segments)
# - cv2.LSD_REFINE_ADV: aggressive refinement (trims + filters + merges; slower)
REFINE = cv2.LSD_REFINE_STD
# Downscale image by this factor before detection:
# - Range: (0, 1]; lower = faster, but might lose fine lines
# - 0.65 is fast and works well if lines are thick and well-contrasted
# - Raise to 0.8–1.0 if you're missing detail
SCALE = 0.8
# Controls Gaussian blur strength before detection:
# - Effective sigma = sigma_scale * scale
# - Higher smooths out noise but can blur away very thin lines
# - Typical range: 0.6 (sharp) to 1.0+ (smooth)
SIGMA_SCALE = 0.95
# Quantization step for gradient magnitude:
# - Larger value -> ignore low-gradient (faint) edges
# - Smaller -> more sensitive (but may detect text, noise)
# - Range: 1.0 to 10.0
# - For clear line art, 2.0–5.0 is good.
QUANT = 4.0
# Angle tolerance (in degrees) for grouping pixels into a line:
# - Smaller = stricter (e.g. only perfectly straight), larger = more tolerant
# - Raise to 25–30 if you want to allow diagonal walls or sketchy lines
# - Lower to 5–10 if you're enforcing only H/V alignment (e.g. grid layout)
ANG_TH = 30
# Minimum segment length (in log scale):
# - log_eps = -log10(min_length_pixels)
# - 0 -> allow 1px segments (i.e. no filtering)
# - 1 -> minimum 10px segments, 2 -> minimum 100px
# - Leave at 0 unless you're getting lots of noise
LOG_EPS = 0
# Line density threshold:
# - Ratio of aligned edge pixels to total segment pixels
# - Higher = more confident lines only (e.g. clean walls)
# - Lower = tolerate broken/partial lines (e.g. faded scans)
# - Typical values:
#     - 0.7–0.9 -> very clean lines only
#     - 0.4–0.6 -> tolerant of broken edges
DENSITY_TH = 0.5
# Number of histogram bins used internally (do not usually change)
# - 1024 is default and sufficient for most cases
N_BINS = 1024

# Snap distance for 2 snap points to be considered close enough to merge.
SNAP_DISTANCE = 6
# Minimum distance between two snap points to be considered non-overlapping.
MIN_INTERSECTION_DIST = 2
# Tolerance for tie-breaking when deciding to merge with another half or just hit a wall.
TIE_BREAK_TOLERANCE = 2
# Size of the hit box for snap points.
SNAP_HIT_BOX_SIZE = 10
# Tolerance for considering two snap points as overlapping in pixels.
EXTRA_CLEARANCE = 3
# Tolerance for angle snapping in degrees.
ANGLE_TOLERANCE_BUCKET = 5
# Tolerance for deciding if an angle is close to 90 degrees.
ANGLE_TOLERANCE_90 = 0.1
# Tolerance for rectangle comparison in pixels, if smaller than 1, no tolerance is applied.
R_TOLERANCE = 4

# Squared values for distance calculations to avoid sqrt.
SNAP_DISTANCE_SQ = SNAP_DISTANCE * SNAP_DISTANCE
MIN_INTERSECTION_DIST_SQ = MIN_INTERSECTION_DIST * MIN_INTERSECTION_DIST

# Canonical angles for snapping, in degrees. Other angles can be generated if calculated angles differ from these
# by more than ANGLE_TOLERANCE_BUCKET.
CANONICAL_ANGLES = (0, 90, -90, 45, -45, 30, -30, 60, -60)

# Small epsilon value for floating point comparisons.
EPS = 1e-2


def ray_line_intersection(origin: Tuple[float, float], direction: tuple, seg_start: Tuple[float, float],
                          seg_end: Tuple[float, float]) -> float | None:
# def ray_line_intersection(
#     origin: Tuple[float, float],
#     direction: Union[Tuple[float, float], List[float]],
#     seg_start: Tuple[float, float],
#     seg_end: Tuple[float, float]
# ) -> Union[float, None]:
    """
    Compute intersection of a ray and a line segment.

    Parameters:
    - origin      (tuple): (ox, oy) start point of the ray
    - direction   (tuple): (dx, dy) unit or scaled direction vector of the ray
    - seg_start   (tuple): (sx, sy) first endpoint of the segment
    - seg_end     (tuple): (ex, ey) second endpoint of the segment

    Returns:
    - t (float): ray parameter where intersection occurs (point = origin + t*direction),
                 provided t >= 0 and the intersection lies within the segment [0 <= u <= 1].
    - None: if lines are parallel, intersection is behind the ray origin, or falls outside the segment.
    """
    ox, oy = origin
    dx, dy = direction
    sx, sy = seg_start
    ex, ey = seg_end

    vx, vy = vec_sub((ex, ey), (sx, sy))
    denominator = dx * vy - dy * vx
    if abs(denominator) < 1e-6:
        return None

    diff_x, diff_y = vec_sub((sx, sy), (ox, oy))
    t = (diff_x * vy - diff_y * vx) / denominator
    if t < 0:
        return None

    if abs(vx) > abs(vy):
        u = (ox + t * dx - sx) / vx
    elif abs(vy) > 1e-6:
        u = (oy + t * dy - sy) / vy
    else:
        return None

    return t if 0 <= u <= 1 else None


def validate_vector(v: tuple | list, name: str = "vector") -> None:
# def validate_vector(v: Union[Tuple, List], name: str = "vector") -> None:
    """
    Validate that the input is a tuple or list of two numeric values (int or float).

    Parameters:
    - v (tuple | list): The input vector to validate.
    - name (str): The name of the vector for error messages.
    """
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
        raise ValueError(f"{name} must be a tuple or list of two numeric values.")


def validate_int(s: int | float, name: str = "scalar") -> None:
# def validate_int(s: Union[int, float], name: str = "scalar") -> None:
    """
    Validate that the input is an integer or a float that can be converted to an integer.

    Parameters:
    - s (int | float): The input scalar to validate.
    - name (str): The name of the scalar for error messages.
    """
    if not isinstance(s, (int, float)) and s != int(s):
        raise ValueError(f"{name} must be integer.")


def vec_add(a: tuple | list, b: tuple | list) -> tuple:
# def vec_add(a: Union[Tuple, List], b: Union[Tuple, List]) -> Tuple[float, float]:
    """
    Add two 2D vectors.

    Parameters:
    - a (tuple/list): First vector as a list or tuple of two numbers.
    - b (tuple/list): Second vector as a list or tuple of two numbers.
    Returns:
    - tuple: Resulting vector as a tuple of two numbers.
    """
    validate_vector(a, "a")
    validate_vector(b, "b")
    ax, ay = a
    bx, by = b
    return ax + bx, ay + by


def vec_sub(a: tuple | list, b: tuple | list) -> tuple:
# def vec_sub(a: Union[Tuple, List], b: Union[Tuple, List]) -> Tuple[float, float]:
    """
    Subtract two 2D vectors.

    Parameters:
    - a (tuple/list): First vector as a list or tuple of two numbers.
    - b (tuple/list): Second vector as a list or tuple of two numbers.
    Returns:
    - tuple: Resulting vector as a tuple of two numbers.
    """
    validate_vector(a, "a")
    validate_vector(b, "b")
    ax, ay = a
    bx, by = b
    return ax - bx, ay - by


def vec_scale(v: tuple | list, s: int | float) -> tuple:
# def vec_scale(v: Union[Tuple, List], s: Union[int, float]) -> Tuple[float, float]:
    """
    Scale a 2D vector by an integer scalar.

    Parameters:
    - v (tuple/list): Vector to scale, as a list or tuple of two numbers.
    - s (int | float): Scalar to scale the vector by.
    Returns:
    - tuple: Scaled vector as a tuple of two numbers.
    """
    validate_vector(v, "v")
    validate_int(s, "s")
    x, y = v
    return x * s, y * s


def vec_dot(a: tuple | list, b: tuple | list) -> float:
# def vec_dot(a: Union[Tuple, List], b: Union[Tuple, List]) -> float:
    """
    Compute the dot product of two 2D vectors.

    Parameters:
    - a (tuple/list): First vector as a list or tuple of two numbers.
    - b (tuple/list): Second vector as a list or tuple of two numbers.
    Returns:
    - float: Dot product of the two vectors.
    """
    validate_vector(a, "a")
    validate_vector(b, "b")
    ax, ay = a
    bx, by = b
    return ax * bx + ay * by


def vec_length_sq(v: tuple | list) -> float:
# def vec_length_sq(v: Union[Tuple, List]) -> float:
    """
    Compute the squared length of a 2D vector.

    Parameters:
    - v (tuple/list): Vector as a list or tuple of two numbers.
    Returns:
    - float: Squared length of the vector.
    """
    validate_vector(v, "v")
    x, y = v
    return x * x + y * y


def vec_normalize(v: tuple | list) -> Tuple[float, float]:
# def vec_normalize(v: Union[Tuple, List]) -> Tuple[float, float]:
    """
    Normalize a 2D vector to unit length.

    Parameters:
    - v (tuple/list): Vector to normalize, as a list or tuple of two numbers.
    Returns:
    - tuple: Normalized vector as a tuple of two numbers (x, y).
    """
    validate_vector(v, "v")
    length = math.hypot(*v)
    if length < EPS:
        return 0.0, 0.0
    inv = 1.0 / length
    x, y = v
    return x * inv, y * inv


def oriented_angle(v1: tuple | list, v2: tuple | list) -> float:
# def oriented_angle(v1: Union[Tuple, List], v2: Union[Tuple, List]) -> float:
    """
    Compute the oriented angle between two 2D vectors in radians.

    Parameters:
    - v1 (tuple/list): First vector as a list or tuple of two numbers.
    - v2 (tuple/list): Second vector as a list or tuple of two numbers.
    Returns:
    - float: Oriented angle in radians, in the range [0, 2pi).
    """
    validate_vector(v1, "v1")
    validate_vector(v2, "v2")
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = vec_dot(v1, v2)
    angle = math.atan2(cross, dot)
    return angle if angle >= 0 else angle + 2 * math.pi


def find(parent: list | dict, i: int | tuple) -> int:
# def find(parent: Union[List, dict], i: Union[int, Tuple]) -> Union[int, Tuple]:
    """
    Returns the root of element `i` in a union-find structure with path compression.

    Parameters:
    - parent (list | dict): Maps elements to their parent.
    - i (int | tuple): Element to find the root of.

    Returns:
    - int: Root representative of `i`.
    """
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(parent: list | dict, a: int | tuple, b: int | tuple) -> None:
# def union(parent: Union[List, dict], a: Union[int, Tuple], b: Union[int, Tuple]) -> None:
    """
    Unites two elements `a` and `b` in a union-find structure.

    Parameters:
    - parent (list): Maps elements to their parent.
    - a (int): First element to unite.
    - b (int): Second element to unite.
    """
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def round_away(v: float, n_comp: float) -> int:
    """
    Rounds the value `v` away from zero based on the sign of `n_comp`.

    If `n_comp` is non-negative, rounds down; if positive, rounds up.

    Parameters:
    - v (float): The value to round.
    - n_comp (float): The component to determine the rounding direction.
    Returns:
    - int: The rounded value.
    """
    return math.floor(v) if n_comp >= 0 else math.ceil(v)


def round_toward(v: float, n_comp: float) -> int:
    """
    Rounds the value `v` toward zero based on the sign of `n_comp`.

    If `n_comp` is non-negative, rounds up; if negative, rounds down.

    Parameters:
    - v (float): The value to round.
    - n_comp (float): The component to determine the rounding direction.
    Returns:
    - int: The rounded value.
    """
    return math.ceil(v) if n_comp >= 0 else math.floor(v)


def choose_rep_angle(angles: List[float]) -> float:
    """
    Compute circular mean of angles, snap to nearest canonical angle within tolerance, otherwise return mean angle.

    Parameters:
    - angles (List[float]): List of angles in degrees.
    Returns:
    - float: The representative angle, snapped to the nearest canonical angle if within tolerance,
    otherwise the mean angle.
    """
    # Circular mean
    rad = [math.radians(a) for a in angles]
    sin_sum = sum(math.sin(r) for r in rad)
    cos_sum = sum(math.cos(r) for r in rad)
    mean_deg = math.degrees(math.atan2(sin_sum, cos_sum))
    if mean_deg > 90:
        mean_deg -= 180
    elif mean_deg < -90:
        mean_deg += 180
    # Snap to canonical
    for ca in CANONICAL_ANGLES:
        if abs(mean_deg - ca) <= ANGLE_TOLERANCE_BUCKET:
            return ca
    return mean_deg


def project_to_axis(endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
                    theta: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Project endpoints onto direction theta and recenter to axis-aligned segment.

    Parameters:
    - endpoints (Tuple[Tuple[float, float], Tuple[float, float]]): The endpoints of the segment as
    tuples (x0, y0), (x1, y1).
    - theta (float): The angle in radians to project onto.
    Returns:
    - Tuple[Tuple[float, float], Tuple[float, float]]: The new endpoints of the segment after projection.
    """
    d = (math.cos(theta), math.sin(theta))
    (x0, y0), (x1, y1) = endpoints
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    t0 = (x0 - mid_x) * d[0] + (y0 - mid_y) * d[1]
    t1 = (x1 - mid_x) * d[0] + (y1 - mid_y) * d[1]
    t_min, t_max = sorted((t0, t1))
    return (mid_x + t_min * d[0], mid_y + t_min * d[1]), (mid_x + t_max * d[0], mid_y + t_max * d[1])


def sample_normal(p1: Tuple[float, float],
                  p2: Tuple[float, float],
                  theta: float,
                  free_mask,  # from _prepare_eroded_mask()
                  clearance: int,
                  img_w: int,
                  img_h: int) -> Tuple[float, float]:
    """
    Sample the original free_mask one pixel beyond the clearance in both
    candidate normal directions; pick the normal pointing toward the side
    with more free‐space.

    Parameters:
    - p1 (Tuple[float, float]): First endpoint of the segment.
    - p2 (Tuple[float, float]): Second endpoint of the segment.
    - theta (float): The angle in radians of the wall segment.
    - free_mask: A 2D array representing the free space mask.
    - clearance (int): Clearance distance in pixels.
    - img_w (int): Width of the image.
    - img_h (int): Height of the image.
    Returns:
    - Tuple[float, float]: The sampled normal vector as a tuple (nx, ny).
    """
    # the two perpendicular directions to the wall
    cand1 = (-math.sin(theta), math.cos(theta))
    cand2 = (math.sin(theta), -math.cos(theta))

    # unit vector along the segment
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L < EPS:
        return cand1
    ux, uy = dx / L, dy / L

    # sample every ~max(10,2*clearance) pixels
    interval = max(10, 2 * clearance)
    n_samp = max(1, int(L / interval))
    # offset distance from the wall: clearance+1 px
    offset = clearance + 1

    sum1 = sum2 = 0
    cnt1 = cnt2 = 0

    for i in range(1, n_samp + 1):
        t = i / (n_samp + 1)
        cx = p1[0] + ux * L * t
        cy = p1[1] + uy * L * t

        # side 1
        sx1 = int(round(cx + cand1[0] * offset))
        sy1 = int(round(cy + cand1[1] * offset))
        if 0 <= sx1 < img_w and 0 <= sy1 < img_h:
            sum1 += free_mask[sy1, sx1]
            cnt1 += 1

        # side 2
        sx2 = int(round(cx + cand2[0] * offset))
        sy2 = int(round(cy + cand2[1] * offset))
        if 0 <= sx2 < img_w and 0 <= sy2 < img_h:
            sum2 += free_mask[sy2, sx2]
            cnt2 += 1

    avg1 = sum1 / cnt1 if cnt1 else -1
    avg2 = sum2 / cnt2 if cnt2 else -1

    return cand1 if avg1 >= avg2 else cand2


def get_xy(pt: tuple | list | object) -> Tuple[float | int, float | int]:
# def get_xy(pt: Union[tuple, list, object]) -> Tuple[Union[float, int], Union[float, int]]:
    """
    Accept either an (x,y) tuple/list of two numbers, or an object with .x and .y.
    Returns a clean (x, y) tuple of floats.

    Parameters:
    - pt (tuple | list | object): The point to convert, either as a tuple/list of two numbers or an object with
    .x and .y attributes.
    Returns:
    - Tuple[float, float]: A tuple of floats representing the x and y coordinates of the point.
    """
    if hasattr(pt, 'x') and hasattr(pt, 'y'):
        return float(pt.x), float(pt.y)
    if isinstance(pt, (tuple, list)) and len(pt) == 2 and all(isinstance(c, (int, float)) for c in pt):
        return float(pt[0]), float(pt[1])
    raise ValueError("Point must be a tuple of two numbers or have .x and .y attributes")
