import math
import os

import cv2
from matplotlib import patches, pyplot as plt
from matplotlib.patches import Rectangle as MplRect

from snappoint import HalfSnapPoint, FullSnapPoint, DoubleSnapPoint
from graph import Graph

from typing import Union

import numpy as np

class Map:
    """
    A class to handle the processing and visualization of a floor plan image.
    """

    def __init__(self, filename: str = None, threshold: int = 250, image: np.ndarray = None) -> None:
        """
        Initializes the Map object with an image file and a threshold for processing.

        Parameters:
        - filename: Path to the image file to be processed.
        - threshold: Threshold value for binarizing the image (default is 250).
        - image: A numpy grayscale image array to use directly (instead of loading from disk).
        """
        # print("[Map] Constructed new LineMap instance.")
        if image is not None:
            # If an image is provided directly
            self.image = image
            self.grid_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
            return

        if not isinstance(filename, str) or not filename:
            raise ValueError("Filename must be a non-empty string.")

        if not os.path.exists(filename):
            raise ValueError(f"File not found: {filename}")

        self.filename = filename

        self.image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Failed to load image: {filename}")

        _, self.grid_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)

    def print(self, filename: str = 'output', extension: str = 'svg', save: bool | int = False, show: bool | int = True,
              debug: bool | int = False, mode: (list, str) = 'rectangles') -> None:
    # def print(self, filename: str = 'output', extension: str = 'svg', save: Union[bool, int] = False, show: Union[bool, int] = True,
            #   debug: Union[bool, int] = False, mode: (list, str) = 'rectangles') -> None:
        """
        Draws the processed image with various features like rectangles, lines, snap points, and hit boxes.

        Parameters:
        - filename: Base name for the output file (default is 'output').
        - extension: File format for saving the output (default is 'svg').
        - save: Whether to save the output to a file (default is False).
        - show: Whether to display the output using matplotlib (default is True).
        - debug: Whether to print debug information (default is False).
        - mode: Feature modes to draw. Can be a single string or a list of strings.
                Valid options are 'rectangles', 'lines', 'snap_points', and 'hit_boxes'.
        """
        if self.grid_image is None:
            raise ValueError("No image loaded.")

        modes = [mode] if isinstance(mode, str) else mode
        valid_modes = {'rectangles', 'lines', 'snap_points', 'hit_boxes'}
        invalid_modes = [m for m in modes if m not in valid_modes]
        if invalid_modes:
            raise ValueError(f"Invalid mode(s): {invalid_modes}. Choose from {valid_modes}.")

        order = {"rectangles": 0, "lines": 1, "snap_points": 2, "hit_boxes": 3}
        modes = sorted(modes, key=lambda x: order[x])

        fig, ax = plt.subplots(
            figsize=(self.grid_image.shape[1] / 100, self.grid_image.shape[0] / 100),
            dpi=100
        )
        plt.axis('off')

        ax.imshow(self.grid_image, cmap='gray', vmin=0, vmax=255, origin='upper', interpolation='none')

        for m in modes:
            try:
                getattr(self, f"_draw_{m}")(ax, debug)
            except AttributeError:
                pass

        if save:
            os.makedirs('out', exist_ok=True)
            plt.savefig(f'out/{filename}.{extension}', bbox_inches='tight', pad_inches=0, format=extension)
        if show:
            plt.show()
        plt.close(fig)

    def _draw_rectangles(self, ax: any, debug: bool = False) -> None:
        """
        Draws rectangles on the given axes.

        Parameters:
        - ax: The axes on which to draw the rectangles.
        - debug: If True, prints debug information about the rectangles (default is False).
        """
        if hasattr(self, 'rectangles'):
            if debug:
                print(f"Number of rectangles: {len(self.rectangles)}")
                for i, rect in enumerate(self.rectangles.nodes()):
                    print(
                        f"Rectangle {i + 1}: Center={rect.center}, Half-extents={rect.half_extents}, Angle={rect.angle}")

            for rect in self.rectangles.nodes():
                shifted = [(x + 0.5, y + 0.5) for x, y in rect.corners]
                ax.add_patch(plt.Polygon(shifted,
                                         closed=True,
                                         edgecolor='red',
                                         facecolor='red',
                                         alpha=0.5))

    def _draw_lines(self, ax: any, debug: bool = False) -> None:
        """
        Draws lines and their normals on the given axes.

        Parameters:
        - ax: The axes on which to draw the lines.
        - debug: If True, prints debug information about the lines (default is False).
        """
        if hasattr(self, 'lines') and self.lines:
            if debug:
                print(f"Number of straightened lines: {len(self.lines)}")
                for i, line in enumerate(self.lines):
                    print(f"Line {i + 1}: {line}")
            for line in self.lines:
                x1, y1 = line.p1
                x2, y2 = line.p2
                ax.add_line(plt.Line2D([x1, x2], [y1, y2], color='lime', linewidth=2))
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                arrow_length = 20
                arrow_end = (
                    mid_x + arrow_length * line.normal[0],
                    mid_y + arrow_length * line.normal[1]
                )
                ax.add_patch(patches.FancyArrowPatch(
                    (mid_x, mid_y), arrow_end,
                    arrowstyle='-|>', color='red', linewidth=2,
                    mutation_scale=15, shrinkA=0, shrinkB=0
                ))
        elif hasattr(self, 'segments_data') and self.segments_data:
            if debug:
                print(f"Number of raw detected line segments: {len(self.segments_data)}")
            for seg in self.segments_data:
                endpoints = seg.get('endpoints')
                if endpoints:
                    (x1, y1), (x2, y2) = endpoints
                    ax.add_line(plt.Line2D([x1, x2], [y1, y2], color='lime', linewidth=2))
        else:
            if debug:
                print("Warning: No lines or segments to display.")

    def _draw_snap_points(self, ax: any, debug: bool = False) -> None:
        """
        Draws snap points on the given axes.

        Parameters:
        - ax: The axes on which to draw the snap points.
        - debug: If True, prints debug information about the snap points (default is False).
        """
        full_points = getattr(self, 'full_snaps', set())
        half_points = getattr(self, 'half_snaps', set())
        double_points = getattr(self, 'double_snaps', set())
        points = full_points.union(half_points).union(double_points)

        if not points:
            if debug:
                print("Warning: No snap points to display.")
            return

        if debug:
            print(f"Number of snap points: {len(points)}")
            for i, pt in enumerate(points):
                if isinstance(pt, FullSnapPoint):
                    print(f"FullSnapPoint {i + 1}: {pt}")
                elif isinstance(pt, HalfSnapPoint):
                    print(f"HalfSnapPoint {i + 1}: {pt}")
                elif isinstance(pt, DoubleSnapPoint):
                    print(f"DoubleSnapPoint {i + 1}: {pt}")

        for pt in points:
            x, y = (pt.x, pt.y) if hasattr(pt, 'x') and hasattr(pt, 'y') else pt
            if isinstance(pt, HalfSnapPoint):
                color = 'purple'
            elif isinstance(pt, FullSnapPoint):
                color = 'blue'
            elif isinstance(pt, DoubleSnapPoint):
                color = 'gold'
            else:
                color = 'black'
            ax.plot(x, y, 'o', markersize=8, color=color)

    def _draw_hit_boxes(self, ax: any, debug: bool = False) -> None:
        """
        Draws hit boxes on the given axes.

        Parameters:
        - ax: The axes on which to draw the hit boxes.
        - debug: If True, prints debug information about the hit boxes (default is False).
        """
        half_points = getattr(self, 'half_snaps', set())

        if not half_points:
            if debug:
                print("Warning: No hitboxes to display.")
            return

        hitbox_count = sum(1 for pt in half_points if getattr(pt, 'hitbox', None))
        if debug:
            print(f"Number of hitboxes: {hitbox_count}")
            for i, pt in enumerate(half_points):
                if isinstance(pt, HalfSnapPoint) and getattr(pt, 'hitbox', None):
                    print(f"Hitbox {i + 1}: {pt.hitbox}")

        for pt in half_points:
            if isinstance(pt, HalfSnapPoint) and getattr(pt, 'hitbox', None):
                (hx1, hy1), (hx2, hy2) = pt.hitbox
                ax.add_line(plt.Line2D([hx1, hx2], [hy1, hy2], color='orange', linewidth=2))

    def draw_graph(self) -> None:
        """
        Draws a graph of overlapping rectangles and saves it as an SVG file.
        """
        graph = getattr(self, 'rectangles', Graph())
        if not graph:
            return

        scale = 0.3

        # set up figure to match image size
        fig, ax = plt.subplots(
            figsize=(self.grid_image.shape[1] / 100,
                     self.grid_image.shape[0] / 100),
            dpi=100
        )

        # draw each scaled‐down rectangle at its center
        for rect in graph.nodes():
            cx, cy = rect.center
            # original full width/height
            orig_w = 2 * rect.half_extents[0]
            orig_h = 2 * rect.half_extents[1]
            # scaled down
            w = orig_w * scale
            h = orig_h * scale
            angle = math.degrees(rect.angle)

            # lower‐left corner for MplRect
            dx, dy = -w / 2, -h / 2
            patch = MplRect(
                (cx + dx, cy + dy),
                w, h,
                angle=angle,
                edgecolor='red',
                facecolor='red',
                alpha=0.5
            )
            ax.add_patch(patch)

        # draw clear black lines between centers of overlapping rectangles
        for u_rect, v_rect in graph.edges():
            x1, y1 = u_rect.center
            x2, y2 = v_rect.center
            ax.plot([x1, x2], [y1, y2],
                    color='black', linewidth=1.0, alpha=0.8)

        # finalize axes
        ax.set_aspect('equal')
        height, width = self.grid_image.shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        ax.set_title("Rectangle-Overlap Graph")

        # save as SVG
        os.makedirs('out', exist_ok=True)
        fig.savefig(
            'out/graph.svg',
            bbox_inches='tight',
            pad_inches=0,
            format='svg'
        )

        # show & clean up
        plt.show()
        plt.close(fig)
