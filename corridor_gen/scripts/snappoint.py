from __future__ import annotations

import math
from typing import Self, TYPE_CHECKING
# from typing import TYPE_CHECKING, TypeVar

# T = TypeVar("T", bound="SnapPoint")

if TYPE_CHECKING:
    # avoid circular import during runtime
    from .line import Line

from util import validate_int, vec_normalize, oriented_angle, EPS


class SnapPoint:
    """
    Base class for snap points in a grid system.
    """

    def __init__(self, x: int | float, y: int | float) -> None:
        """
        Initializes a SnapPoint at the given coordinates (x, y).

        Parameters:
        - x: The x-coordinate of the snap point.
        - y: The y-coordinate of the snap point.
        """
        validate_int(x, "x")
        validate_int(y, "y")
        self.x = x
        self.y = y
        self.lines = set()
        self.lines1 = set()
        self.lines2 = set()
        self.hit_line = None
        self.hitbox = None
        self.used = False
        self.face = None

    def get_angle(self) -> float:
        """
        Returns the angle of the snap point relative to its associated lines.

        This method should be overridden in subclasses.

        Returns:
        - float: The angle in radians, or None if not applicable.
        """
        raise NotImplementedError

    def follow(self, line: Line) -> Self:
    # def follow(self: T, line: Line) -> T:
        """
        Determines which snap point to follow based on the given line.

        This method should be overridden in subclasses.

        Parameters:
        - line: The line connected to this snap point.
        Returns:
        - SnapPoint: The other snap point on the same line, or None if not applicable.
        """
        raise NotImplementedError

    def is_right(self) -> bool:
        """
        Checks if the snap point is 90 degrees to its associated lines.

        Returns:
        - bool: True if the angle is approximately 90 degrees, False otherwise.
        """
        angle = self.get_angle()
        return angle is not None and abs(angle - math.pi / 2) < EPS

    def _build_repr(self, extra_parts: list) -> str:
        """
        Builds a string representation of the SnapPoint instance.

        Parameters:
        - extra_parts: Additional parts to include in the representation.
        Returns:
        - str: A string representation of the SnapPoint instance.
        """
        parts = [f"{self.__class__.__name__}({self.x}, {self.y}"]
        if self.hitbox is not None:
            parts.append(f"hitbox={self.hitbox}")
        parts.extend(extra_parts)
        return ", ".join(parts) + ")"

    def __repr__(self) -> str:
        """
        Returns a string representation of the SnapPoint instance.

        Returns:
        - str: A string representation of the SnapPoint instance.
        """
        return self._build_repr([])


class DoubleSnapPoint(SnapPoint):
    """
    Represents two full snap points that are overlapping.
    """

    def __init__(self, x: int | float, y: int | float) -> None:
        """
        Initializes a DoubleSnapPoint at the given coordinates (x, y).

        Parameters:
        - x: The x-coordinate of the snap point.
        - y: The y-coordinate of the snap point.
        """
        super().__init__(x, y)
        self.dir1 = False
        self.dir2 = False

    def get_angle(self) -> float:
        """
        Returns the angle of the snap point in radians.

        For double snap points, this is always pi/2.

        Returns:
        - float: The angle in radians, which is always pi/2 for double snap points.
        """
        return math.pi / 2

    def follow(self, line: Line) -> Self | None:
        """
        Given a line connected to self, returns the other snap point on that same line.

        Parameters:
        - line: The line connected to this snap point.
        Returns:
        - SnapPoint: The other snap point on the same line, or None if not applicable.
        """
        for group in (self.lines1, self.lines2):
            if line in group and len(group) == 2:
                sp1, sp2 = line.snap_points
                return sp1 if sp2 is self else sp2
        return None

    def __repr__(self) -> str:
        """
        Returns a string representation of the DoubleSnapPoint instance.

        Returns:
        - str: A string representation of the DoubleSnapPoint instance.
        """

        def fmt(groups):
            return ", ".join(f"Line({l.p1} -> {l.p2})" for l in groups)

        extra = [f"lines1=[{fmt(self.lines1)}]", f"lines2=[{fmt(self.lines2)}]"]
        return self._build_repr(extra)


class FullSnapPoint(SnapPoint):
    """
    Represents a snap point that is connected to two lines.
    """

    def get_angle(self) -> float | None:
        """
        Returns the angle between the two lines connected to this snap point in radians.

        Returns:
        - float: The angle in radians between the normals of the two lines, or None if not applicable.
        """
        if len(self.lines) != 2:
            return None
        line1, line2 = self.lines
        raw = oriented_angle(vec_normalize(line1.normal), vec_normalize(line2.normal))
        min_ang = raw if raw <= math.pi else 2 * math.pi - raw
        return math.pi - min_ang

    def follow(self, line: Line) -> Self | None:
        """
        Given a line connected to self, returns the other snap point on that same line.

        Parameters:
        - line: The line connected to this snap point.
        Returns:
        - SnapPoint: The other snap point on the same line, or None if not applicable.
        """
        if len(self.lines) == 2 and line in self.lines:
            sp1, sp2 = line.snap_points
            return sp1 if sp2 is self else sp2
        return None

    def __repr__(self) -> str:
        """
        Returns a string representation of the FullSnapPoint instance.

        Returns:
        - str: A string representation of the FullSnapPoint instance.
        """

        def fmt(groups):
            return ", ".join(f"Line({l.p1} -> {l.p2})" for l in groups)

        extra = [f"lines=[{fmt(self.lines)}]"]
        return self._build_repr(extra)


class HalfSnapPoint(SnapPoint):
    """
    Represents a snap point that is connected to a single line.
    """

    def __init__(self, x: int | float, y: int | float) -> None:
        """
        Initializes a HalfSnapPoint at the given coordinates (x, y).

        Parameters:
        - x: The x-coordinate of the snap point.
        - y: The y-coordinate of the snap point.
        """
        super().__init__(x, y)
        self.sister = None
        self.preserve = False
        self.old_x = x
        self.old_y = y

    def compute_hitbox(self, size: float) -> None:
        """
        Computes the hitbox for this snap point based on the given hitbox size.

        Parameters:
        - size: The size of the hitbox, which must be a positive number.
        """
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError("Hitbox size must be a positive number.")
        if not self.lines:
            raise RuntimeError("Cannot compute hitbox: no associated lines.")
        n = next(iter(self.lines)).normal
        half = size / 2
        self.hitbox = ((self.x - half * n[0], self.y - half * n[1]),
                       (self.x + half * n[0], self.y + half * n[1]))

    def get_angle(self) -> float | None:
        """
        Returns the angle between the normal of the first line and the normal of the hit line.

        Returns:
        - float: The angle in radians between the normals of the first line and the hit line, or None if not applicable.
        """
        if not self.lines or self.hit_line is None or len(self.lines) != 1:
            return None
        n1 = vec_normalize(next(iter(self.lines)).normal)
        n2 = vec_normalize(self.hit_line.normal)
        raw = oriented_angle(n1, n2)
        min_ang = raw if raw <= math.pi else 2 * math.pi - raw
        return math.pi - min_ang

    def follow(self, line: Line) -> Self | None:
        """
        If the given line is connected to this snap point, returns the other snap point on that same line.
        If the given line is the hit line, returns the snap on that hit_line which is in the direction of the normal
        of the connected line.

        Parameters:
        - line: The line connected to this snap point.
        Returns:
        - SnapPoint: The other snap point on the same line, or None if not applicable.
        """
        if len(self.lines) != 1:
            return None
        base = next(iter(self.lines))
        if line is not base or self.hit_line is None:
            return None
        sp1, sp2 = self.hit_line.snap_points
        if None in (sp1, sp2):
            return None
        nx, ny = self.hit_line.get_normal()
        d = [(sp.x - self.x) * nx + (sp.y - self.y) * ny for sp in (sp1, sp2)]
        return (sp1, sp2)[d.index(max(d))]

    def __repr__(self) -> str:
        """
        Returns a string representation of the HalfSnapPoint instance.

        Returns:
        - str: A string representation of the HalfSnapPoint instance.
        """

        def fmt(groups):
            return ", ".join(f"Line({l.p1} -> {l.p2})" for l in groups)

        extra = [f"lines=[{fmt(self.lines)}]", f"hit_line={self.hit_line}"]
        return self._build_repr(extra)
