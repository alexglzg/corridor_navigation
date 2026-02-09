import math
from itertools import permutations
from typing import Tuple, Self

from .util import vec_sub, vec_add, vec_dot, get_xy, EPS, CANONICAL_ANGLES, ANGLE_TOLERANCE_BUCKET, R_TOLERANCE


class Rectangle:
    """
    Represents a rectangle defined by three points and a map bounds.
    """

    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int],
                 map_bounds: Tuple[int, int]) -> None:
        """
        Initializes a Rectangle object with three points and map bounds.

        Parameters:
        - p1: First point as a tuple (x, y).
        - p2: Second point as a tuple (x, y).
        - p3: Third point as a tuple (x, y).
        - map_bounds: Tuple (width, height) defining the bounds of the map.
        """

        # validate map bounds
        width, height = map_bounds
        if not (isinstance(width, (int, float)) and width > 0 and isinstance(height, (int, float)) and height > 0):
            raise ValueError("map_bounds must be a tuple of two positive numbers")

        # normalize input
        pts = [get_xy(p) for p in (p1, p2, p3)]

        # find the right-angle corner (a,b,c)
        best = None
        best_diff = float('inf')
        for a, b, c in permutations(pts, 3):
            v1 = vec_sub(a, b)
            v2 = vec_sub(c, b)
            l1, l2 = math.hypot(*v1), math.hypot(*v2)
            if l1 == 0 or l2 == 0:
                continue
            cos_v = vec_dot(v1, v2) / (l1 * l2)
            angle_deg = abs(math.degrees(math.acos(max(-1, min(1, cos_v)))))
            diff = abs(90 - angle_deg)
            if diff < best_diff:
                best_diff = diff
                best = (a, b, c)
        if best is None:
            raise ValueError("Invalid corner input; no right-angle triple found")
        a, b, c = best

        # compute fourth corner
        d = vec_add(a, vec_sub(c, b))
        corners = [a, b, c, d]

        # center
        cx = sum(x for x, _ in corners) / 4
        cy = sum(y for _, y in corners) / 4
        self.center = (cx, cy)

        # sort CCW
        corners.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        self.corners = corners

        # half-extents
        edge = vec_sub(corners[1], corners[0])
        orthogonal = vec_sub(corners[3], corners[0])
        half_w = math.hypot(*edge) / 2
        half_h = math.hypot(*orthogonal) / 2
        self.half_extents = (half_w, half_h)

        # raw angle
        raw_deg = math.degrees(math.atan2(edge[1], edge[0])) % 180
        if raw_deg > 90:
            raw_deg -= 180

        # snap to canonical or round to integer
        snapped = next((ca for ca in CANONICAL_ANGLES if abs(raw_deg - ca) <= ANGLE_TOLERANCE_BUCKET), None)
        final_deg = snapped if snapped is not None else round(raw_deg)
        self.angle = math.radians(final_deg)

        # bounds check
        if not self._within_bounds(width, height, self.center, self.half_extents, self.angle):
            raise ValueError("Rectangle out of bounds")

    @classmethod
    # def from_four_points(cls, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], p4: Tuple[int, int],
    #                  map_bounds: Tuple[int, int], tol: float = EPS) -> 'Rectangle':
    def from_four_points(cls, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], p4: Tuple[int, int],
                        map_bounds: Tuple[int, int], tol: float = EPS) -> Self:
        """
        Creates a Rectangle from four points, ensuring the fourth point matches one of the corners.

        Parameters:
        - p1: First point as a tuple (x, y).
        - p2: Second point as a tuple (x, y).
        - p3: Third point as a tuple (x, y).
        - p4: Fourth point as a tuple (x, y) that should match one of the corners.
        - map_bounds: Tuple (width, height) defining the bounds of the map.
        """
        rect = cls(p1, p2, p3, map_bounds)
        x4, y4 = get_xy(p4)
        for x, y in rect.corners:
            if abs(x - x4) <= tol and abs(y - y4) <= tol:
                return rect
        raise ValueError("Fourth point does not match any corner")

    @staticmethod
    def _within_bounds(width: int, height: int, center: Tuple[float, float], half_extents: Tuple[float, float],
                       angle: float) -> bool:
        """
        Checks if the rectangle defined by center, half_extents, and angle is within the given bounds.

        Parameters:
        - width: Width of the map bounds.
        - height: Height of the map bounds.
        - center: Center of the rectangle as a tuple (cx, cy).
        - half_extents: Half extents of the rectangle as a tuple (hx, hy).
        - angle: Rotation angle of the rectangle in radians.
        Returns:
        - bool: True if the rectangle is within bounds, False otherwise.
        """
        cx, cy = center
        hx, hy = half_extents
        ca, sa = math.cos(angle), math.sin(angle)
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                x = cx + dx * ca - dy * sa
                y = cy + dx * sa + dy * ca
                if not (0 <= x <= width and 0 <= y <= height):
                    return False
        return True

    def get_area(self) -> float:
        """
        Returns the area of the rectangle.

        Returns:
        - float: The area of the rectangle.
        """
        return 4 * self.half_extents[0] * self.half_extents[1]

    def _quantized_key(self) -> Tuple[int, int, int, int, int]:
        """
        Returns a quantized key for the rectangle based on its center, half extents, and angle. This key is used for
        hashing and equality checks, ensuring that rectangles with similar properties are treated as equal. This is
        based on the allowed tolerances R_TOLERANCE and ANGLE_TOLERANCE_BUCKET.

        Returns:
        - Tuple[int, int, int, int, int]: A tuple representing the quantized key of the rectangle.
        """
        # bucket sizes chosen to match your tolerances:
        pixel_bucket = max(2 * R_TOLERANCE, 1.0)
        angle_bucket = 2 * math.radians(ANGLE_TOLERANCE_BUCKET)

        cx, cy = self.center
        hx, hy = self.half_extents
        ang = self.angle

        bx = round(cx / pixel_bucket)
        by = round(cy / pixel_bucket)
        bhx = round(hx / pixel_bucket)
        bhy = round(hy / pixel_bucket)
        ba = round(ang / angle_bucket)

        return bx, by, bhx, bhy, ba

    def __eq__(self, other: Self) -> bool:
    # def __eq__(self, other: 'Rectangle') -> bool:
        """
        Checks if two Rectangle objects are equal based on their quantized keys.

        Parameters:
        - other: Another Rectangle object to compare with.
        Returns:
        - bool: True if the rectangles are equal, False otherwise.
        """
        if not isinstance(other, Rectangle): return False
        return self._quantized_key() == other._quantized_key()

    def __hash__(self) -> int:
        """
        Returns a hash value for the Rectangle object based on its quantized key.

        Returns:
        - int: The hash value of the rectangle.
        """
        return hash(self._quantized_key())

    def __repr__(self) -> str:
        """
        Returns a string representation of the Rectangle object, including its center, half extents, and angle.

        Returns:
        - str: A string representation of the Rectangle object.
        """
        return (f"Rectangle(center=({self.center[0]:.2f}, {self.center[1]:.2f}), "
                f"half_extents=({self.half_extents[0]:.2f}, {self.half_extents[1]:.2f}), "
                f"angle={math.degrees(self.angle):.2f}°)")
