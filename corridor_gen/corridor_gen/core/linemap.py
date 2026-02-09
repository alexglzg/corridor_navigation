import time
from collections import defaultdict
from typing import cast, Dict, Any, Set

from .graph import Graph
from .line import Line
from .map import Map
from .rectangle import Rectangle
from .snappoint import HalfSnapPoint, FullSnapPoint, DoubleSnapPoint
from .util import *

import numpy as np

class LineMap(Map):
    """
    A Map subclass specialized for extracting a representation of a floor plan using overlapping rectangles.

    It detects and processes wall segments, snaps their endpoints to true corners,
    and builds a graph of Rectangle nodes in `self.rectangles`, where each rectangle
    represents one room block.
    """

    def __init__(self, filename: str = None, threshold: int = 250, debug: (bool, int) = False, image: np.ndarray = None, resolution: float = 0.05, robot_clearance: float = 0.3) -> None:
        """
        Initialize a LineMap for extracting rectangular room layouts.

        Parameters:
          filename (str): path to a binary (black‐and‐white) floor-plan image file.
          threshold (int): grayscale threshold for binarization in the base Map constructor.
          image (np.ndarray): alternative image input (grayscale) instead of filename.

        Behavior:
          1. Validate inputs.
          2. Call Map.__init__ to load and threshold the image into self.grid_image.
          3. Ensure the binary image is present.
          4. Initialize empty containers for each processing stage.
        """
        # 1) Input validation
        # if not isinstance(filename, str) or not filename:
        #     raise ValueError("filename must be a non-empty string")
        if not isinstance(threshold, int):
            raise ValueError("threshold must be an integer")
        if not isinstance(debug, (bool, int)):
            raise ValueError("debug must be a boolean or integer")
        # self.debug = bool(debug)

        # if debug:
        #     print(f"Initializing LineMap with filename='{filename}', threshold={threshold}, debug={self.debug}")

        if filename is None and image is None:
            raise ValueError("Either 'filename' or 'image' must be provided.")
        
        self.debug = bool(debug)
        if self.debug:
            print(f"Initializing LineMap with threshold={threshold}, debug={self.debug}")

        # 2) Load and binarize image via base class
        super().__init__(filename=filename, threshold=threshold, image=image)

        # 3) Confirm that grid_image was created successfully
        if not hasattr(self, 'grid_image') or self.grid_image is None:
            raise ValueError("Failed to load image; grid_image is missing or None")

        # 4) Prepare data structures for each pipeline stage
        self.segments_data = []  # raw LSD segments
        self.lines = set()  # cleaned, straightened Line objects
        self.full_snaps = set()  # convex-corner points
        self.half_snaps = set()  # concave-corner points
        self.double_snaps = set()  # 4-way junction points (double full snaps)
        self.rectangles = Graph()  # final room rectangles as a connectivity graph
        self.obstacles = dict()  # set of obstacle faces (non-room areas)

        self.min_rect_size = (robot_clearance / resolution) / 2 # minimum rectangle size in pixels

        if self.debug:
            print("LineMap initialized successfully with empty data structures.")

    def detect_lines(self) -> None:
        """
        Detect raw wall segments in the binary floor-plan using OpenCV's Line Segment Detector (LSD).

        Populates:
        -self.segments_data (list of dict):
            - 'seg'       : the raw segment array from LSD
            - 'angle'     : normalized angle in (–90°, 90°] of the segment
            - 'endpoints' : tuple of endpoint coordinates ((x0, y0), (x1, y1))
        """
        if self.debug:
            print(f"Creating Line Segment Detector with parameters:\n"
                  f"  refine={REFINE}, scale={SCALE}, sigma_scale={SIGMA_SCALE}, "
                  f"quant={QUANT}, ang_th={ANG_TH}, log_eps={LOG_EPS}, "
                  f"density_th={DENSITY_TH}, n_bins={N_BINS}")
        try:
            lsd = cv2.createLineSegmentDetector(
                refine=REFINE,
                scale=SCALE,
                sigma_scale=SIGMA_SCALE,
                quant=QUANT,
                ang_th=ANG_TH,
                log_eps=LOG_EPS,
                density_th=DENSITY_TH,
                n_bins=N_BINS
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Line Segment Detector: {e}")

        if self.debug:
            print("Line Segment Detector created successfully, starting line detection...")
        try:
            raw_segments = lsd.detect(self.grid_image)[0]
        except Exception as e:
            raise RuntimeError(f"LSD detection failed: {e}")

        if raw_segments is None:
            raise RuntimeError("LSD detection returned None")

        if self.debug:
            print(f"Detected {len(raw_segments)} raw line segments.")

        # Filter and record each valid segment
        for segment in raw_segments:
            # skip degenerate or too-short entries
            if segment is None or segment.size < 4:
                continue
            x0, y0, x1, y1 = segment[0]
            dx, dy = x1 - x0, y1 - y0
            if dx * dx + dy * dy < 1:
                continue

            # compute angle in (–90°, 90°]
            angle = math.degrees(math.atan2(dy, dx))
            angle = ((angle + 90) % 180) - 90

            self.segments_data.append({
                'seg': segment,
                'angle': angle,
                'endpoints': ((x0, y0), (x1, y1))
            })
        if not self.segments_data:
            raise RuntimeError("No valid line segments detected; check input image quality or parameters.")
        if self.debug:
            print(f"Filtered down to {len(self.segments_data)} valid segments after processing.")

    def _prepare_eroded_mask(self) -> Tuple[int, Any]:
        """
        Build a binary free-space mask (walls=0, free=1) and return it
        along with the integer clearance.

        Returns:
        - clearance (int): the clearance value used for inward shifting.
        - free_mask: binary mask of free space.
        """
        if not hasattr(self, 'grid_image') or self.grid_image is None:
            raise RuntimeError("No grid image available; call detect_lines() first.")
        if self.grid_image.ndim != 2 or self.grid_image.dtype != 'uint8':
            raise ValueError("grid_image must be a 2D binary image of type uint8")

        # build free-space mask: walls=0, free=1
        _, free_mask = cv2.threshold(self.grid_image, 0, 1, cv2.THRESH_BINARY)
        clearance = int(EXTRA_CLEARANCE)
        return clearance, free_mask

    def _cluster_and_represent_angles(self) -> Tuple[List[int], Dict[int, float]]:
        """
        Cluster segment angles by grouping any two segments whose angles differ
        by at most ANGLE_TOLERANCE_BUCKET (in degrees), then pick a representative
        for each cluster (snapping to canonical angles when within tolerance).

        Returns:
        - labels (List[int]): cluster labels for each segment.
        - rep_angles (Dict[int, float]): representative angle for each cluster.
        """
        if not self.segments_data:
            raise RuntimeError("No segments detected. Call detect_line_segments() first.")
        if self.debug:
            print("Clustering segment angles (±{:.0f}°) and finding representative angles..."
                  .format(ANGLE_TOLERANCE_BUCKET))

        # 1) gather all segment angles in degrees
        angles = [item['angle'] for item in self.segments_data]  # already in (−90, +90]

        # 2) build similarity graph: connect i↔j if angle difference ≤ tol
        tol = ANGLE_TOLERANCE_BUCKET
        n = len(angles)
        graph = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(angles[i] - angles[j])
                # account for wrap-around at +/-90 -> difference never exceeds 180 here
                if diff > tol:
                    continue
                graph[i].append(j)
                graph[j].append(i)

        # 3) find connected components via DFS/union-find
        labels = [-1] * n
        comp_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            stack = [i]
            while stack:
                u = stack.pop()
                if labels[u] == -1:
                    labels[u] = comp_id
                    stack.extend(graph[u])
            comp_id += 1

        # 4) group original angles by cluster
        from collections import defaultdict
        clusters = defaultdict(list)
        for lbl, ang in zip(labels, angles):
            clusters[lbl].append(ang)

        # 5) pick representative for each cluster
        rep_angles = {}
        for lbl, angles in clusters.items():
            rep = choose_rep_angle(angles)
            rep_angles[lbl] = rep

        if self.debug:
            print(f"Found {len(clusters)} clusters with representative angles.")
            for lbl, rep in rep_angles.items():
                print(f"  Cluster {lbl}: representative angle = {rep:.2f}°")

        return labels, rep_angles

    def _build_lines(self, labels: List[int], rep_angles: Dict[int, float], eroded_free, clearance: int) -> Set[Line]:
        """
        Process each raw segment: straighten, drop too-short, auto-sample normals,
        then shift endpoints. Returns set of Line objects.

        Parameters:
        - labels (List[int]): cluster labels for each segment.
        - rep_angles (Dict[int, float]): representative angle for each cluster.
        - eroded_free: binary mask of free space after morphological erosion.
        - clearance (int): clearance value for inward shifting.
        Returns:
        - detected (Set[Line]): set of Line objects created from segments.
        """
        if not self.segments_data:
            raise RuntimeError("No segments detected. Call detect_line_segments() first.")
        if self.debug:
            print("Building lines from raw segments...")
        detected = set()
        h, w = self.grid_image.shape
        # min_len = max(2 * clearance, 2)
        min_len = max(1 * clearance, 1)

        for lbl, item in zip(labels, self.segments_data):
            theta = math.radians(rep_angles[lbl])
            # Project raw endpoints to canonical line
            p1, p2 = project_to_axis(item['endpoints'], theta)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            # Choose inward normal via sampled mask
            normal = sample_normal(p1, p2, theta, eroded_free, clearance, w, h)
            # Shift endpoints by clearance
            p1_adj = (round(p1[0] + normal[0] * clearance), round(p1[1] + normal[1] * clearance))
            p2_adj = (round(p2[0] + normal[0] * clearance), round(p2[1] + normal[1] * clearance))
            try:
                detected.add(Line(p1_adj, p2_adj, normal))
            except Exception as e:
                raise RuntimeError(f"Line creation failed: {e}")
        if not detected:
            raise RuntimeError("No valid lines created; check segment data and parameters.")
        if self.debug:
            print(f"Built {len(detected)} lines from raw segments after processing.")
        return detected

    def straighten_lines(self) -> None:
        """
        Cluster raw segments by orientation, snap them to canonical angles,
        then shift them inward by uniform clearance using morphological erosion.
        Produces a set of Line objects in self.lines.
        """
        if not self.segments_data:
            raise RuntimeError(
                "No segments detected. Call detect_line_segments() first."
            )
        if self.debug:
            print("Straightening and processing detected line segments...")

        # Prepare masks for inward shifting
        clearance, eroded_free = self._prepare_eroded_mask()

        # Cluster segment orientations and pick representative angles
        labels, rep_angles = self._cluster_and_represent_angles()

        # Straighten each segment, drop degenerate ones, and shift
        self.lines = self._build_lines(labels, rep_angles, eroded_free, clearance)

        if not self.lines:
            raise RuntimeError("No valid lines created after processing segments.")
        if self.debug:
            print(f"Straightened lines: {len(self.lines)} valid lines created.")

    def snap_endpoints(self) -> None:
        """
        Snap each Line's endpoints to exact corner points by extending or shrinking along their original
        direction so that walls close precisely and force integer coordinates according to the rounding rules.

        After this, self.full_snaps and self.half_snaps hold the corner points,
        and each Line in self.lines has endpoints moved exactly to its assigned snap points.
        """
        if not self.lines:
            raise RuntimeError(
                "No lines to snap. Call detect_line_segments() and adjust_line_segments() first."
            )

        if self.debug:
            print("Snapping line endpoints to exact corner points...")

        h, w = self.grid_image.shape

        # 1) Save original endpoints for later projection
        orig_ends = {ln: (ln.p1, ln.p2) for ln in self.lines}

        # 2) Collect all endpoints + references
        endpoints = []
        endpoint_refs = []
        for ln in self.lines:
            endpoints.append(ln.p1)
            endpoint_refs.append((ln, 0))
            endpoints.append(ln.p2)
            endpoint_refs.append((ln, 1))

        # 3) Clustering via hash-grid + union find
        snap_r = math.sqrt(SNAP_DISTANCE_SQ)
        cell_sz = max(1.0, snap_r)
        buckets = defaultdict(list)
        for idx, (x, y) in enumerate(endpoints):
            buckets[(int(x // cell_sz), int(y // cell_sz))].append(idx)

        parent = list(range(len(endpoints)))

        def _try_union(ia_var, ib_var) -> None:
            if endpoint_refs[ia_var][0] is endpoint_refs[ib_var][0]:
                return

            dx_var = endpoints[ia_var][0] - endpoints[ib_var][0]
            dy_var = endpoints[ia_var][1] - endpoints[ib_var][1]
            if dx_var * dx_var + dy_var * dy_var <= SNAP_DISTANCE_SQ:
                union(parent, ia_var, ib_var)

        neigh = ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))  # examine each pair once
        for (gx, gy), indexes in buckets.items():
            # (a) intra-bucket pairs
            for a in range(len(indexes)):
                ia = indexes[a]
                for b in range(a + 1, len(indexes)):
                    _try_union(ia, indexes[b])

            # (b) pairs with forward-neighbour buckets
            for dx, dy in neigh[1:]:
                nbr = buckets.get((gx + dx, gy + dy))
                if not nbr:
                    continue
                for ia in indexes:
                    for ib in nbr:
                        _try_union(ia, ib)

        # gather clusters
        cluster_map: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(endpoints)):
            cluster_map[find(parent, idx)].append(idx)

        clusters, cluster_refs = [], []
        for members in cluster_map.values():
            cx = sum(endpoints[i][0] for i in members) / len(members)
            cy = sum(endpoints[i][1] for i in members) / len(members)
            clusters.append((cx, cy))
            cluster_refs.append([endpoint_refs[i] for i in members])

        # helper: intersection of infinite lines
        def intersect(p1, p2, q1, q2):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = q1
            x4, y4 = q2
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < EPS:
                return None
            ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
            iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
            return ix, iy

        full_snaps, half_snaps, snap_map = set(), set(), {}

        def signed_angle(u, v):
            return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

        # 4) classify clusters into FullSnapPoint or HalfSnapPoint
        for (cx, cy), refs in zip(clusters, cluster_refs):
            # try full-snap for two distinct lines forming a convex corner
            if len(refs) == 2 and refs[0][0] is not refs[1][0]:
                (ln1, i1), (ln2, i2) = refs
                ln1, ln2 = cast(Line, ln1), cast(Line, ln2)
                n1 = vec_normalize(ln1.get_normal())
                n2 = vec_normalize(ln2.get_normal())

                def direction_away(ln_var, idx_var):
                    o1, o2 = orig_ends[ln_var]
                    p_other = o2 if idx_var == 0 else o1
                    dx_var, dy_var = p_other[0] - cx, p_other[1] - cy
                    L_var = math.hypot(dx_var, dy_var)
                    return (dx_var / L_var, dy_var / L_var) if L_var > EPS else None

                v1, v2 = direction_away(ln1, i1), direction_away(ln2, i2)
                if v1 and v2:
                    bx, by = n1[0] + n2[0], n1[1] + n2[1]
                    bm = math.hypot(bx, by)
                    if bm > EPS:
                        b = (bx / bm, by / bm)
                        ang = signed_angle(v1, v2)
                        small = abs(ang)
                        large = 2 * math.pi - small
                        ang_b = signed_angle(v1, b)
                        in_small = ((ang >= 0 and 0 <= ang_b <= ang) or
                                    (ang < 0 and ang <= ang_b <= 0))
                        interior = small if in_small else large
                        if interior < math.pi - EPS:
                            pt = intersect(ln1.p1, ln1.p2, ln2.p1, ln2.p2)
                            if pt is not None:
                                sp = FullSnapPoint(pt[0], pt[1])
                                full_snaps.add(sp)
                                for ln, idx in refs:
                                    snap_map[(ln, idx)] = sp
                                    sp.lines.add(ln)
                                continue

            # half-snap at infinite-line intersection
            if len(refs) == 2 and refs[0][0] is not refs[1][0]:
                (ln1, _), (ln2, _) = refs
                ln1, ln2 = cast(Line, ln1), cast(Line, ln2)
                pt = intersect(ln1.p1, ln1.p2, ln2.p1, ln2.p2)
                if pt is not None:
                    new_snaps = []
                    for ln, idx in refs:
                        sp = HalfSnapPoint(pt[0], pt[1])
                        half_snaps.add(sp)
                        snap_map[(ln, idx)] = sp
                        sp.lines.add(ln)
                        new_snaps.append(sp)

                    if len(new_snaps) == 2:
                        a, b = new_snaps
                        a.sister = b
                        b.sister = a
                    continue

            # fallback: single-end cluster
            for ln, idx in refs:
                sp = HalfSnapPoint(cx, cy)
                half_snaps.add(sp)
                snap_map[(ln, idx)] = sp
                sp.lines.add(ln)

        self.full_snaps, self.half_snaps = full_snaps, half_snaps

        # 5) project endpoints, round to integers by rules, and move the lines
        for ln in self.lines:
            sp1, sp2 = snap_map[(ln, 0)], snap_map[(ln, 1)]
            (o1x, o1y), (o2x, o2y) = orig_ends[ln]
            dx, dy = o2x - o1x, o2y - o1y
            L = math.hypot(dx, dy)
            if L < EPS:
                continue

            def project(sp_var):
                ux, uy = dx / L, dy / L
                vx, vy = sp_var.x - o1x, sp_var.y - o1y
                t = vx * ux + vy * uy
                return o1x + ux * t, o1y + uy * t

            # reproject exactly onto line
            sp1.x, sp1.y = project(sp1)
            sp2.x, sp2.y = project(sp2)

            # get inward normal for rounding
            nx, ny = ln.get_normal()

            # enforce integer snap-point coords
            for sp in (sp1, sp2):
                xi = round_away(sp.x, nx)
                yi = round_away(sp.y, ny)
                # if that pixel is closed (1), shrink instead
                if 0 <= xi < w and 0 <= yi < h and self.grid_image[yi, xi] == 1:
                    xi = round_toward(sp.x, nx)
                    yi = round_toward(sp.y, ny)
                sp.x, sp.y = xi, yi

            try:
                ln.move(sp1, sp2)
            except Exception as e:
                raise RuntimeError(f"Failed to move line: {e}")

        # 6) build hit-boxes around half-snaps for extension logic
        for sp in self.half_snaps:
            try:
                sp.compute_hitbox(SNAP_HIT_BOX_SIZE)
            except Exception as e:
                raise RuntimeError(f"Failed to compute hitbox for SnapPoint: {e}")

        if self.debug:
            print(f"Snapped endpoints: {len(self.full_snaps)} full snaps, "
                  f"{len(self.half_snaps)} half snaps created.")

    def extend_half_snap_points(self) -> None:
        """
        Resolve each concave (half) corner by extending its line
        until it meets the nearest wall or another half-snap point.
        This helps to ensure that all corners are properly connected
        and that the floor plan is fully represented without gaps.
        """
        if not self.half_snaps:
            if self.debug:
                print("No half-snap points to extend; skipping extension.")
            return
        if self.debug:
            print("Extending half-snap points to resolve concave corners...")
        # Work from a copy, then rebuild self.half_snaps
        half_snaps_work = set(self.half_snaps)
        self.half_snaps.clear()

        # 1) Bucket by orientation
        def _orient_key(n):
            return round(abs(n[0]), 4), round(abs(n[1]), 4)

        orient_buckets = defaultdict(list)
        for sp in half_snaps_work:
            if sp.lines:
                orient_buckets[_orient_key(next(iter(sp.lines)).normal)].append(sp)

        # 2) Process each concave snap
        for current_sp in list(half_snaps_work):
            if len(current_sp.lines) != 1:
                continue

            current_line = next(iter(current_sp.lines))
            orig_normal = current_line.normal
            origin = (current_sp.x, current_sp.y)

            # ray direction outward from the nearer endpoint
            d1 = vec_length_sq(vec_sub(origin, current_line.p1))
            d2 = vec_length_sq(vec_sub(origin, current_line.p2))
            ray_dir = vec_sub(current_line.p2, origin) if d1 < d2 else vec_sub(current_line.p1, origin)
            ray_dir = vec_scale(vec_normalize(ray_dir), -1)

            # 2a) nearest parallel half-snap
            best_half_t = float('inf')
            best_half = None
            for cand in orient_buckets[_orient_key(orig_normal)]:
                if cand is current_sp or not cand.lines:
                    continue
                ln = next(iter(cand.lines))
                ndx = ln.normal[0] - orig_normal[0]
                ndy = ln.normal[1] - orig_normal[1]
                pdx = ln.normal[0] + orig_normal[0]
                pdy = ln.normal[1] + orig_normal[1]
                if min(ndx * ndx + ndy * ndy, pdx * pdx + pdy * pdy) > EPS:
                    continue
                dv = vec_sub((cand.x, cand.y), origin)
                t = vec_dot(dv, ray_dir)
                if t <= 0:
                    continue
                perp = vec_sub(dv, vec_scale(ray_dir, t))
                hb = current_sp.hitbox
                if hb is None:
                    raise RuntimeError("Half-snap point has no hitbox.")
                # noinspection PyUnresolvedReferences
                hb_vec = vec_sub(hb[1], hb[0])
                if vec_length_sq(perp) <= vec_length_sq(hb_vec) / 4 and t < best_half_t:
                    best_half_t, best_half = t, cand

            # 2b) nearest wall
            best_line_t = float('inf')
            best_line = None
            for ln in self.lines:
                if ln is current_line or getattr(ln, 'extension', False):
                    continue
                if vec_dot(ln.normal, ray_dir) > -0.5:
                    continue
                for p1, p2 in ln.orig_segments:
                    t = ray_line_intersection(origin, ray_dir, p1, p2)
                    if t and 0 < t < best_line_t:
                        ip = vec_add(origin, vec_scale(ray_dir, t))
                        if vec_length_sq(vec_sub(ip, origin)) > MIN_INTERSECTION_DIST_SQ:
                            best_line_t, best_line = t, ln

            # 3) pick target
            choose_half = (
                    best_half is not None and
                    (best_line is None or
                     best_half_t < best_line_t or
                     abs(best_half_t - best_line_t) < TIE_BREAK_TOLERANCE)
            )

            # 4a) merge into another half-snap
            if choose_half:
                target_sp = best_half
                t = best_half_t
                # preserve both sisters of the merging snaps
                for sp in (target_sp, current_sp):
                    sis = getattr(sp, 'sister', None)
                    if sis is not None:
                        sis.preserve = True
                half_snaps_work.discard(target_sp)

                ln_cand = next(iter(target_sp.lines))
                other_sp_cand = ln_cand.snap_points[0] if ln_cand.snap_points[1] is target_sp else ln_cand.snap_points[
                    1]
                other_sp_cur = (current_line.snap_points[0]
                                if current_line.snap_points[1] is current_sp
                                else current_line.snap_points[1])

                ix, iy = origin[0] + ray_dir[0] * t, origin[1] + ray_dir[1] * t
                current_sp.x, current_sp.y = ix, iy

                try:
                    current_line.extend(other_sp_cur, other_sp_cand,
                                        merge_orig_segments=ln_cand.orig_segments)
                except Exception as e:
                    raise RuntimeError("Failed merging half-snap: " + str(e))

                if ln_cand in self.lines:
                    self.lines.remove(ln_cand)
                    for sp in ln_cand.snap_points:
                        if sp:
                            sp.lines.discard(ln_cand)

            # 4b) extend to a real wall
            elif best_line is not None:
                t = best_line_t
                # noinspection PyUnresolvedReferences
                ix, iy = origin[0] + ray_dir[0] * t, origin[1] + ray_dir[1] * t
                current_sp.x, current_sp.y = ix, iy
                current_sp.hit_line = best_line
                best_line.hit_half_snaps.add(current_sp)
                try:
                    current_line.extend(*current_line.snap_points)
                except Exception as e:
                    raise RuntimeError("Failed extending to wall: " + str(e))
                self.half_snaps.add(current_sp)

            else:
                continue  # nothing hit

            # 5) snap to integers, drop hitbox
            current_sp.x = int(round(current_sp.x))
            current_sp.y = int(round(current_sp.y))
            current_sp.hitbox = None

        # 6) re-add any sister-preserved points
        for sp in half_snaps_work:
            if getattr(sp, "preserve", False):
                self.half_snaps.add(sp)

        if self.debug:
            print(f"Extended half-snap points: {len(self.half_snaps)} half snaps after extension.")

    def identify_faces(self) -> None:
        """
        Walk every boundary‐cycle (face) in the snap‐graph, classify
        it by whether its wall‐normals point inward (part of the building) or outward
        (obstacles), then merge any obstacle snaps that share a wall or sister link
        into single obstacle groups. Assign face=0 to building, 1+ to each obstacle,
        populate self.obstacles (mapping face_id->set of original (old_x,old_y) coords),
        and finally remove all obstacle snaps/lines (except preserved half-snaps).
        """
        if not (self.double_snaps or self.full_snaps or self.half_snaps):
            raise RuntimeError("No snaps available. Call snap_endpoints() first.")
        if self.debug:
            print("Identifying faces in the snap graph...")

        # 1) collect all snaps
        all_snaps = set(self.double_snaps) | set(self.full_snaps) | set(self.half_snaps)
        unvisited = set(all_snaps)
        cycles = []

        max_iterations = len(all_snaps) * 10
        iteration_count = 0

        # helper to walk around a boundary cycle
        def _next_in_cycle(sp, prev_sp_var):
            for ln_var in sp.lines:
                other = ln_var.snap_points[0] if ln_var.snap_points[1] is sp else ln_var.snap_points[1]
                if other is not prev_sp_var and other is not None:
                    return other
            return None

        # 2) extract each cycle of snaps
        while unvisited and iteration_count < max_iterations:
            iteration_count += 1
            start = unvisited.pop()
            cycle = [start]
            prev_sp, curr_sp = None, start
            visited_in_cycle = {start}
            while True:
                nbr = _next_in_cycle(curr_sp, prev_sp)
                if nbr is None or nbr is start:
                    break
                if nbr in visited_in_cycle:
                    print(f"Warning: Detected cycle within traversal at {nbr}")
                cycle.append(nbr)
                unvisited.discard(nbr)
                prev_sp, curr_sp = curr_sp, nbr

            if len(cycle) <= 3:
                cycles.append(cycle)

        if iteration_count >= max_iterations:
            raise RuntimeError("Cycle extraction exceeded maximum iterations; check for infinite loops or malformed graph.")

        # 3) initial classification: any cycle whose outward–normals > inward -> obstacle
        obs_old_coords = set()
        for cycle in cycles:
            # centroid
            cx = sum(sp.x for sp in cycle) / len(cycle)
            cy = sum(sp.y for sp in cycle) / len(cycle)
            inward = outward = 0
            n = len(cycle)
            for i, sp in enumerate(cycle):
                sp2 = cycle[(i + 1) % n]
                # shared wall?
                ln = next((L for L in sp.lines if sp2 in L.snap_points), None)
                if ln is None:
                    outward += 1
                else:
                    mx, my = (sp.x + sp2.x) * 0.5, (sp.y + sp2.y) * 0.5
                    vx, vy = cx - mx, cy - my
                    if ln.normal[0] * vx + ln.normal[1] * vy > 0:
                        inward += 1
                    else:
                        outward += 1
            if outward > inward:
                # obstacle cycle -> record its snaps' original coords
                for sp in cycle:
                    ox = getattr(sp, 'old_x', sp.x)
                    oy = getattr(sp, 'old_y', sp.y)
                    obs_old_coords.add((ox, oy))

        # 4) remove any preserved half-snap from removal
        preserved = {
            (getattr(sp, 'old_x', sp.x), getattr(sp, 'old_y', sp.y))
            for sp in all_snaps
            if getattr(sp, 'preserve', False)
        }
        obs_old_coords -= preserved

        # 5) union‐find to merge obstacle coords that connect via sister or shared line
        parent = {coord: coord for coord in obs_old_coords}

        # for each snap in obs_old_coords, union with sister (if obstacle) and line‐neighbors
        for sp in all_snaps:
            ox = getattr(sp, 'old_x', sp.x)
            oy = getattr(sp, 'old_y', sp.y)
            coord = (ox, oy)
            if coord not in parent:
                continue
            # sister link
            sis = getattr(sp, 'sister', None)
            if sis is not None:
                ox2 = getattr(sis, 'old_x', sis.x)
                oy2 = getattr(sis, 'old_y', sis.y)
                coord2 = (ox2, oy2)
                if coord2 in parent:
                    union(parent, coord, coord2)
            # shared‐wall link
            for ln in sp.lines:
                for nbr in ln.snap_points:
                    if nbr is sp or nbr is None:
                        continue
                    ox2 = getattr(nbr, 'old_x', nbr.x)
                    oy2 = getattr(nbr, 'old_y', nbr.y)
                    coord2 = (ox2, oy2)
                    if coord2 in parent:
                        union(parent, coord, coord2)

        # 6) collect each disjoint set as one obstacle
        from collections import defaultdict
        comps = defaultdict(set)
        for coord in obs_old_coords:
            comps[find(parent, coord)].add(coord)

        # remap into consecutive face IDs
        self.obstacles = {}
        coord_to_face = {}
        for fid, comp_coords in enumerate(comps.values(), start=1):
            self.obstacles[fid] = comp_coords
            for coord in comp_coords:
                coord_to_face[coord] = fid

        # 7) assign final face IDs on snaps (0 = building; >0 = obstacle)
        for sp in all_snaps:
            ox = getattr(sp, 'old_x', sp.x)
            oy = getattr(sp, 'old_y', sp.y)
            sp.face = coord_to_face.get((ox, oy), 0)

        # 8) prune away obstacle snaps
        def _is_obs_snap(sp):
            ox_var = getattr(sp, 'old_x', sp.x)
            oy_var = getattr(sp, 'old_y', sp.y)
            return (ox_var, oy_var) in obs_old_coords

        self.double_snaps = {sp for sp in self.double_snaps if not _is_obs_snap(sp)}
        self.full_snaps = {sp for sp in self.full_snaps if not _is_obs_snap(sp)}
        self.half_snaps = {sp for sp in self.half_snaps if not _is_obs_snap(sp)}

        # 9) drop any wall‐line touching a removed obstacle snap (except if that snap was preserved)
        def _touches_obs_line(ln_var):
            for sp in ln_var.snap_points:
                if sp is None:
                    continue
                ox_var = getattr(sp, 'old_x', sp.x)
                oy_var = getattr(sp, 'old_y', sp.y)
                if (ox_var, oy_var) in obs_old_coords and not getattr(sp, 'preserve', False):
                    return True
            return False

        self.lines = {ln for ln in self.lines if not _touches_obs_line(ln)}

        if self.debug:
            print(f"Identified {len(self.obstacles)} obstacle regions, "
                  f"{len(self.double_snaps)} double snaps, "
                  f"{len(self.full_snaps)} full snaps, "
                  f"{len(self.half_snaps)} half snaps after pruning.")

    def extend_full_snap_points(self) -> None:
        """
        For each obtuse-angle FullSnapPoint:
        - Identify its two incident walls.
        - Cast one 90° extension from each.
        - Replace the FullSnapPoint with a DoubleSnapPoint whose lines1/lines2
        are both sets of exactly two lines (wall + its extension).
        """
        if not self.full_snaps:
            if self.debug:
                print("No full-snap points to extend; skipping extension.")
            return
        if self.debug:
            print("Extending full-snap points to resolve obtuse corners...")

        active = [ln for ln in self.lines if not getattr(ln, "extension", False)]
        seg_map = {ln: ln.orig_segments for ln in active}

        for full_snap in list(self.full_snaps):
            walls = [ln for ln in full_snap.lines if ln in active]
            if len(walls) != 2:
                continue
            wall1, wall2 = walls

            # 1) Check if the angle between the two walls is obtuse
            n1 = vec_normalize(wall1.get_normal())
            n2 = vec_normalize(wall2.get_normal())
            raw = oriented_angle(n1, n2)
            interior = math.pi - (raw if raw <= math.pi else 2 * math.pi - raw)
            if interior <= math.pi / 2 + ANGLE_TOLERANCE_90:
                continue

            rx, ry = full_snap.x, full_snap.y
            ext_info = []

            # 2) For each wall, cast a ray outward to find the nearest line intersection
            for src in (wall1, wall2):
                origin = (rx, ry)
                direction = vec_normalize(src.normal)
                best_d, best_ln = float("inf"), None

                for ln in active:
                    if ln is src or getattr(ln, "extension", False):
                        continue
                    dp = ln.normal[0] * direction[0] + ln.normal[1] * direction[1]
                    if dp > -0.5:
                        continue
                    for p1, p2 in seg_map[ln]:
                        t = ray_line_intersection(origin, direction, p1, p2)
                        if t and 0 < t < best_d and t * t > MIN_INTERSECTION_DIST_SQ:
                            best_d, best_ln = t, ln

                if best_ln is None:
                    ext_info = []
                    break

                fx = rx + direction[0] * best_d
                fy = ry + direction[1] * best_d

                # reuse or create a snap at (fx,fy)
                snap = next(
                    (sp for pool in (self.double_snaps, self.full_snaps, self.half_snaps)
                     for sp in pool
                     if (sp.x - fx) ** 2 + (sp.y - fy) ** 2 <= SNAP_DISTANCE_SQ),
                    None
                )
                if snap is None:
                    snap = HalfSnapPoint(fx, fy)
                    self.half_snaps.add(snap)
                    snap.hit_line = best_ln
                    best_ln.hit_half_snaps.add(snap)

                # pick left/right normal pointing into free space
                left_n = (-direction[1], direction[0])
                right_n = (direction[1], -direction[0])
                other_sp = src.snap_points[1] if src.snap_points[0] is full_snap else src.snap_points[0]
                vx, vy = other_sp.x - rx, other_sp.y - ry
                ext_n = left_n if (left_n[0] * vx + left_n[1] * vy) >= (right_n[0] * vx + right_n[1] * vy) else right_n

                ext_ln = Line((rx, ry), (fx, fy), normal=ext_n)
                ext_ln.extension = True
                self.lines.add(ext_ln)

                ext_info.append((src, ext_ln, snap))

            if len(ext_info) != 2:
                continue

            # 3) Create DoubleSnapPoint with sets for lines1/lines2
            dsp = DoubleSnapPoint(int(round(rx)), int(round(ry)))
            dsp.lines1 = set()
            dsp.lines2 = set()

            # assign original walls + extensions to the two groups
            (src1, ext1, snap1), (src2, ext2, snap2) = ext_info
            dsp.lines1.add(src1)
            dsp.lines1.add(ext1)
            dsp.lines2.add(src2)
            dsp.lines2.add(ext2)

            # replace full_snap with dsp in all lines
            for ln in list(self.lines):
                for i, sp in enumerate(ln.snap_points):
                    if sp is full_snap:
                        ln.snap_points[i] = dsp

            # 4) Wire up each extension line's snap_points
            for src, ext_ln, snap_pt in ext_info:
                ext_ln.snap_points = [None, None]
                if ext_ln.p1 == (dsp.x, dsp.y):
                    ext_ln.snap_points[0] = dsp
                    ext_ln.snap_points[1] = snap_pt
                else:
                    ext_ln.snap_points[1] = dsp
                    ext_ln.snap_points[0] = snap_pt
                # register in snap_pt
                snap_pt.lines.add(ext_ln)

            # 5) finalize
            self.double_snaps.add(dsp)
            self.full_snaps.discard(full_snap)

        if self.debug:
            print(f"Extended full-snap points: {len(self.double_snaps)} double snaps created, "
                  f"{len(self.full_snaps)} full snaps remaining after extension.")

    def generate_rects(self) -> None:
        """
        Identify rectangular rooms and corridors from the snap points.
        This method processes full snaps, half snaps, and double snaps to create
        rectangles representing rooms or corridors in the map.

        The rectangles are stored in self.rectangles, which is a graph-like structure.
        In this step only the nodes are added, edges will be created later.

        Obstacles are not processed here, they are handled in cut_obstacles().
        """
        if not self.double_snaps and not self.full_snaps and not self.half_snaps:
            raise RuntimeError(
                "No snaps available. Call snap_endpoints() first."
            )
        if self.debug:
            print("Generating rectangles from snap points...")
        # full_pts = set(self.full_snaps)
        full_pts = sorted(list(self.full_snaps), key=lambda p: (p.x, p.y)) 
        half_pts = sorted(self.half_snaps, key=lambda p: p.x + p.y)
        # double_pts = set(self.double_snaps)
        double_pts = sorted(list(self.double_snaps), key=lambda p: (p.x, p.y))
        map_bounds = (self.grid_image.shape[1], self.grid_image.shape[0])

        def three_corner_candidates(cand_1, cand_2, cand_3, cand_4, cand_5):
            trips = (
                (cand_2, cand_1, cand_3),
                (cand_1, cand_2, cand_4),
                (cand_1, cand_3, cand_5),
            )
            out = []
            for p, q, r in trips:
                if not q.is_right():
                    continue
                try:
                    out.append(Rectangle(p, q, r, map_bounds))
                except ValueError:
                    pass
            return out

        # Double‐snap rooms
        # while double_pts:
        #     ds = double_pts.pop()
        while double_pts:
            ds = double_pts.pop(0)  # Changed to pop(0) for consistency
            for idx, lines in ((1, ds.lines1), (2, ds.lines2)):
                if getattr(ds, f"dir{idx}"):
                    continue
                setattr(ds, f"dir{idx}", True)
                try:
                    wa, wb = lines
                except ValueError:
                    continue

                s2 = ds.follow(wa)
                s3 = ds.follow(wb)
                if not s2 or not s3:
                    continue

                wc = wa.follow(s2)
                s4 = s2.follow(wc)
                wd = wb.follow(s3)
                s5 = s3.follow(wd)

                room = None
                if s4 == s5:
                    try:
                        room = Rectangle.from_four_points(ds, s2, s3, s4, map_bounds)
                    except ValueError:
                        pass
                else:
                    candidates = three_corner_candidates(ds, s2, s3, s4, s5)
                    if candidates:
                        room = min(candidates, key=lambda r: r.get_area())

                if room:
                    # If the room half extents are large enough, add them
                    if room.half_extents[0] >= self.min_rect_size and \
                            room.half_extents[1] >= self.min_rect_size:
                        self.rectangles.add_node(room)

        # Full‐snap rooms
        # while full_pts:
        #     fs = full_pts.pop()
        while full_pts:
            fs = full_pts.pop(0)  # Changed to pop(0) for consistency
            walls = list(fs.lines)
            if len(walls) != 2:
                continue
            wa, wb = walls

            s2 = fs.follow(wa)
            s3 = fs.follow(wb)
            if not s2 or not s3 or not s2.is_right() or not s3.is_right():
                continue

            wc = wa.follow(s2)
            s4 = s2.follow(wc)
            wd = wb.follow(s3)
            s5 = s3.follow(wd)

            room = None
            if s4 == s5:
                try:
                    room = Rectangle.from_four_points(fs, s2, s3, s4, map_bounds)
                except ValueError:
                    pass

            if not room:
                candidates = []
                for a, b, c in ((s2, fs, s3), (fs, s2, s4), (fs, s3, s5)):
                    if not b.is_right():
                        continue
                    try:
                        candidates.append((Rectangle(a, b, c, map_bounds), (a, b, c)))
                    except ValueError:
                        pass
                if candidates:
                    room, _ = min(candidates, key=lambda x: x[0].get_area())

            if room:
                # If the room half extents are large enough, add them
                if room.half_extents[0] >= self.min_rect_size and \
                        room.half_extents[1] >= self.min_rect_size:
                    self.rectangles.add_node(room)

        # Half‐snap corridors
        while half_pts:
            hp = half_pts.pop(0)
            if len(hp.lines) != 1 or hp.hit_line is None:
                continue
            ext = next(iter(hp.lines))
            p0, p1 = ext.snap_points
            if not (isinstance(p0, HalfSnapPoint) and isinstance(p1, HalfSnapPoint)):
                continue
            n_ext = vec_normalize(ext.normal)

            wall = hp.hit_line
            candidates = [c for c in wall.hit_half_snaps if c is not hp and c in half_pts]
            if not candidates:
                continue

            limit = float('inf')
            towards = []
            for c in candidates:
                ce = next(iter(c.lines))
                n_c = vec_normalize(ce.normal)
                dx, dy = c.x - hp.x, c.y - hp.y
                d = dx * n_ext[0] + dy * n_ext[1]

                if vec_dot(n_c, n_ext) > EPS and 0 < d < limit:
                    limit = d
                if vec_dot(n_c, n_ext) < EPS and abs(n_c[0] + n_ext[0] + n_c[1] + n_ext[1]) < 2 * EPS:
                    towards.append((c, d))

            group = [(c, d) for c, d in towards if 0 < d < limit]
            if not group:
                continue

            far_c, _ = max(group, key=lambda x: x[1])

            # identify the four segment endpoints
            hp_op = p1 if p0 is hp else p0
            far_ln = next(iter(far_c.lines))
            far_op = far_ln.snap_points[1] if far_ln.snap_points[0] is far_c else far_ln.snap_points[0]

            # fit largest rectangle between hp->hp_op and far_c->far_op
            P1, P2, Q1, Q2 = hp, hp_op, far_c, far_op
            vx, vy = P2.x - P1.x, P2.y - P1.y
            Lp = math.hypot(vx, vy)
            if Lp < EPS:
                continue
            ux, uy = vx / Lp, vy / Lp
            nx, ny = -uy, ux
            vqx, vqy = Q1.x - P1.x, Q1.y - P1.y
            if vec_dot((vqx, vqy), (nx, ny)) < 0:
                nx, ny = -nx, -ny
            delta = vec_dot((vqx, vqy), (nx, ny))
            toff = vec_dot((vqx, vqy), (ux, uy))
            Lq = math.hypot(Q2.x - Q1.x, Q2.y - Q1.y)
            s_min = max(0.0, toff)
            s_max = min(Lp, Lq + toff)
            if s_max <= s_min + EPS:
                continue
            Ax, Ay = P1.x + s_min * ux, P1.y + s_min * uy
            Bx, By = P1.x + s_max * ux, P1.y + s_max * uy
            Cx, Cy = Bx + delta * nx, By + delta * ny
            Dx, Dy = Ax + delta * nx, Ay + delta * ny

            class _P:
                def __init__(self, x, y):
                    self.x, self.y = x, y

            A, B, C, D = _P(Ax, Ay), _P(Bx, By), _P(Cx, Cy), _P(Dx, Dy)
            try:
                rect = Rectangle.from_four_points(
                    get_xy(A), get_xy(B), get_xy(C), get_xy(D),
                    map_bounds
                )
                # if the rectangle half extents are large enough, add it
                if rect.half_extents[0] >= self.min_rect_size and \
                        rect.half_extents[1] >= self.min_rect_size:
                    self.rectangles.add_node(rect)
            except ValueError:
                continue

            for sp in (hp_op, far_op, far_c):
                if sp in half_pts:
                    half_pts.remove(sp)

        if self.debug:
            print(f"Generated rectangles: {len(self.rectangles)} rectangles created from snap points.")
            for rect in self.rectangles.nodes():
                print(f"Rectangle: {rect}")

    def cut_obstacles(self) -> None:
        """
        For each rectangle, carve out each obstacle’s bounding box
        (in the room’s rotated frame) and split the leftover into up to
        four overlapping rectangular fragments that each span the full
        width or full height of the original room.
        """
        map_h, map_w = self.grid_image.shape
        map_bounds = (map_w, map_h)

        new_rooms = []
        to_remove = []

        for room in list(self.rectangles.nodes()):
            fragments = [room]

            for obs_pts in self.obstacles.values():
                next_frags = []
                for frag in fragments:
                    # transform obstacle points into frag-local coords
                    local = []
                    fx, fy = frag.center
                    fhx, fhy = frag.half_extents
                    fa = frag.angle
                    ca, sa = math.cos(fa), math.sin(fa)
                    for ox, oy in obs_pts:
                        dx, dy = ox - fx, oy - fy
                        x_loc = dx * ca + dy * sa
                        y_loc = -dx * sa + dy * ca
                        local.append((x_loc, y_loc))

                    xs = [x for x, _ in local]
                    ys = [y for _, y in local]
                    xmin = max(min(xs), -fhx)
                    xmax = min(max(xs), +fhx)
                    ymin = max(min(ys), -fhy)
                    ymax = min(max(ys), +fhy)

                    # no overlap -> keep fragment
                    if xmax <= xmin or ymax <= ymin:
                        next_frags.append(frag)
                        continue

                    # carve into four overlapping slabs
                    slabs = [
                        (-fhx, xmin, -fhy, +fhy),  # left: full height
                        (xmax, +fhx, -fhy, +fhy),  # right: full height
                        (-fhx, +fhx, -fhy, ymin),  # bottom: full width
                        (-fhx, +fhx, ymax, +fhy),  # top:    full width
                    ]

                    for x0, x1, y0, y1 in slabs:
                        w = x1 - x0
                        h = y1 - y0
                        if w <= 0 or h <= 0:
                            continue

                        # local corners CCW
                        local_corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                        world = []
                        for x_loc, y_loc in local_corners:
                            wx = fx + (x_loc * ca - y_loc * sa)
                            wy = fy + (x_loc * sa + y_loc * ca)
                            world.append((wx, wy))

                        # helper point
                        class P:
                            def __init__(self, xy):
                                self.x, self.y = xy

                        A, B, C, D = P(world[0]), P(world[1]), P(world[2]), P(world[3])
                        try:
                            sub = Rectangle.from_four_points(
                                (A.x, A.y), (B.x, B.y), (C.x, C.y), (D.x, D.y),
                                map_bounds
                            )
                            next_frags.append(sub)
                        except ValueError:
                            pass

                fragments = next_frags

            # if carved, replace original
            if len(fragments) > 1 or fragments[0] is not room:
                to_remove.append(room)
                new_rooms.extend(fragments)

        # commit
        for r in to_remove:
            self.rectangles.remove_node(r)
        for r in new_rooms:
            # if the rectangle is large enough, add it
            if r.half_extents[0] >= self.min_rect_size and \
                    r.half_extents[1] >= self.min_rect_size:
                self.rectangles.add_node(r)

    def generate_edges(self) -> None:
        """
        Add an undirected edge between any two room‐rectangles that overlap.
        This method uses the Separating Axis Theorem (SAT) to determine
        whether two rectangles overlap, and it adds edges to the
        `self.rectangles` graph structure to represent these connections.
        """

        def _separating_axis_theorem(corners1, corners2, axes):
            """
            Check if two sets of rectangle corners overlap using the
            Separating Axis Theorem (SAT).
            """
            # project both corner sets onto each axis and check for overlap
            for ax, ay in axes:
                # normalize axis
                length = math.hypot(ax, ay)
                ux, uy = ax / length, ay / length
                # projections for rect1
                proj1 = [px * ux + py * uy for px, py in corners1]
                min1, max1 = min(proj1), max(proj1)
                # projections for rect2
                proj2 = [qx * ux + qy * uy for qx, qy in corners2]
                min2, max2 = min(proj2), max(proj2)
                # if they do not overlap, no intersection
                if max1 < min2 or max2 < min1:
                    return False
            return True

        # clear existing edges
        for u, v in list(self.rectangles.edges()):
            self.rectangles.remove_edge(u, v)

        rooms = list(self.rectangles.nodes())
        n = len(rooms)
        for i in range(n):
            r1 = rooms[i]
            # precompute r1 corners and its two axes
            c1 = r1.corners
            a1 = r1.angle
            cos1, sin1 = math.cos(a1), math.sin(a1)
            axes1 = [
                (cos1, sin1),
                (-sin1, cos1),
            ]

            for j in range(i + 1, n):
                r2 = rooms[j]
                # precompute r2 corners and its axes
                c2 = r2.corners
                a2 = r2.angle
                cos2, sin2 = math.cos(a2), math.sin(a2)
                axes2 = [
                    (cos2, sin2),
                    (-sin2, cos2),
                ]

                # test overlap on all four axes
                if _separating_axis_theorem(c1, c2, axes1 + axes2):
                    self.rectangles.add_edge(r1, r2)

    def process(self) -> None:
        """
        Process the grid image to generate rectangles.
        This method is the main entry point for processing the grid image,
        and it encapsulates the entire snapping and rectangle generation workflow.

        It performs the following steps in sequence:
        1. Detect lines in the grid image.
        2. Straighten the detected lines.
        3. Snap endpoints of the lines to create snap points.
        4. Extend half-snap points to resolve concave corners.
        5. Identify faces (obstacles) in the map.
        6. Extend full-snap points to resolve obtuse corners.
        7. Generate rectangles from the snap points.
        8. Cut obstacles from the rectangles.
        9. Generate edges between overlapping rectangles.
        """

        # Prepare data structures for each pipeline stage
        self.segments_data = []  # raw LSD segments
        self.lines = set()  # cleaned, straightened Line objects
        self.full_snaps = set()  # convex-corner points
        self.half_snaps = set()  # concave-corner points
        self.double_snaps = set()  # 4-way junction points (double full snaps)
        self.rectangles = Graph()  # final room rectangles as a connectivity graph
        self.obstacles = dict()  # set of obstacle faces (non-room areas)

        # Clear any cached attributes that might exist
        for attr in ['hit_half_snaps', 'hit_line', 'sister', 'preserve']:
            if hasattr(self, attr):
                delattr(self, attr)

        t0 = time.time()
        self.detect_lines()
        t1 = time.time()
        self.straighten_lines()
        t2 = time.time()
        self.snap_endpoints()
        t3 = time.time()
        self.extend_half_snap_points()
        t4 = time.time()
        # self.identify_faces()
        # t5 = time.time()
        self.extend_full_snap_points()
        t6 = time.time()
        self.generate_rects()
        t7 = time.time()
        # self.cut_obstacles()
        # t8 = time.time()
        self.generate_edges()
        t9 = time.time()

        print("--------------------------------------------------------------")
        print(f"Processing completed in {(t9 - t0) * 1000:.5f} ms:")
        if self.debug:
            print(f"Line detection: {(t1 - t0) * 1000:.3f} ms.")
            print(f"Line straightening: {(t2 - t1) * 1000:.3f} ms.")
            print(f"Endpoint snapping: {(t3 - t2) * 1000:.3f} ms.")
            print(f"Half-snap extension: {(t4 - t3) * 1000:.3f} ms.")
            # print(f"Face identification: {(t5 - t4) * 1000:.3f} ms.")
            # print(f"Full-snap extension: {(t6 - t5) * 1000:.3f} ms.")
            print(f"Rectangle generation: {(t7 - t6) * 1000:.3f} ms.")
            # print(f"Obstacle cutting: {(t8 - t7) * 1000:.3f} ms.")
            # print(f"Edge generation: {(t9 - t8) * 1000:.3f} ms.")
