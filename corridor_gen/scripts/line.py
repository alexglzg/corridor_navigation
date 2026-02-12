from typing import Tuple, Self
# from typing import Tuple

from snappoint import HalfSnapPoint, SnapPoint
from util import validate_vector, vec_length_sq


class Line:
    """
    Represents a straight line segment, used to model the edge of a wall.

    Each line is defined by two endpoints and a normal vector. The normal points outward,
    toward open space, and is perpendicular to the line segment.
    """

    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], normal: Tuple[float, float]) -> None:
        """
        Initializes a Line object with two endpoints and a normal vector.

        Parameters:
        - p1: First endpoint as a tuple (x, y).
        - p2: Second endpoint as a tuple (x, y).
        - normal: Normal vector as a tuple (nx, ny) pointing outward from the line.
        """
        validate_vector(p1, "p1")
        validate_vector(p2, "p2")
        validate_vector(normal, "normal")

        self.p1 = p1
        self.p2 = p2
        self.normal = normal
        self.orig_segments = [(p1, p2)]
        self.snap_points = [None, None]
        self.extended = False
        self.hit_half_snaps = set()
        self.intersections = set()

    def move(self, sp1: SnapPoint, sp2: SnapPoint) -> None:
        """
        Moves the line to new snap points sp1 and sp2, updating endpoints and linkage.

        Parameters:
        - sp1: First SnapPoint object.
        - sp2: Second SnapPoint object.
        """
        # validate and update snap points
        for idx, sp in enumerate((sp1, sp2), start=1):
            if not (hasattr(sp, "x") and hasattr(sp, "y")):
                raise ValueError(f"sp{idx} must have x and y attributes.")
            if not hasattr(sp, "lines"):
                raise ValueError("SnapPoint objects must have a 'lines' attribute.")
            if isinstance(sp, HalfSnapPoint):
                sp.lines.clear()
            sp.lines.add(self)

        # update endpoints and linkage
        self.p1, self.p2 = (sp1.x, sp1.y), (sp2.x, sp2.y)
        self.snap_points[:] = [sp1, sp2]

    def extend(self, sp1: SnapPoint, sp2: SnapPoint, merge_orig_segments: list = None) -> None:
        """
        Extends the line to new snap points sp1 and sp2. Original segments are merged if provided.

        Parameters:
        - sp1: First SnapPoint object.
        - sp2: Second SnapPoint object.
        - merge_orig_segments: Optional list of original segments to merge into this line.
        """
        # remove this line from source snaps and any overlapping normals
        for src, tgt in ((sp1, sp2), (sp2, sp1)):
            if self in src.lines:
                src.lines.remove(self)
                for other in list(tgt.lines):
                    if other.normal == self.normal:
                        tgt.lines.remove(other)

        self.move(sp1, sp2)
        self.extended = True
        if merge_orig_segments:
            self.orig_segments.extend(merge_orig_segments)

    def length_sq(self) -> float:
        """
        Returns the squared length of the line segment.

        Returns:
        - float: The squared length of the line segment.
        """
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return vec_length_sq((dx, dy))

    def get_normal(self) -> Tuple[float, float]:
        """
        Returns the normal vector of the line segment.

        Returns:
        - Tuple[float, float]: The normal vector (nx, ny) of the line segment.
        """
        if self.normal is None:
            raise ValueError("Normal vector is not set.")
        return self.normal

    def follow(self, snap_point: SnapPoint) -> Self:
    # def follow(self, snap_point: SnapPoint) -> "Line":
        """
        Returns the line that follows the given snap point.

        Parameters:
        - snap_point: A SnapPoint object that contains lines to follow.
        Returns:
        - Line: The line that follows the snap point, or None if no valid line is found.
        """
        lines = snap_point.lines
        count = len(lines)
        # half-snap case
        if count == 1:
            if isinstance(snap_point, HalfSnapPoint) and next(iter(lines)) == self:
                return snap_point.hit_line
            return next(iter(lines))
        # full-snap case
        if count == 2:
            return next(iter(lines - {self}))
        # double-snap case
        if len(snap_point.lines1) == 2 and len(snap_point.lines2) == 2:
            group = snap_point.lines1 if self in snap_point.lines1 else snap_point.lines2
            return next(iter(group - {self}))
        return None

    def __repr__(self) -> str:
        """
        Returns a string representation of the Line object, including endpoints, normal vector,

        Returns:
        - str: A string representation of the Line object.
        """
        # noinspection PyUnresolvedReferences
        snaps = [f"{type(sp).__name__}({sp.x}, {sp.y})" if sp else "None"
                 for sp in self.snap_points]
        return (
            f"Line[{self.p1} -> {self.p2}, normal={self.normal}, "
            f"orig_segments={self.orig_segments}, snaps={snaps}, extended={self.extended}]"
        )
