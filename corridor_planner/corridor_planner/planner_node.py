import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Empty, Float64MultiArray
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from corridor_navigation_interfaces.srv import GetGraph

import networkx as nx
import math
import json
import tf_transformations
import numpy as np
from shapely.geometry import Polygon, Point as ShapelyPoint, LineString
from shapely.affinity import rotate, translate
from shapely.ops import nearest_points
from shapely import STRtree
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set
import time

from nav_msgs.msg import Path
from .core.plan_motion import PlanMotion
import kappa_planner as arena
from kappa_planner import corridor as Corridor

EPS = 1e-9


@dataclass
class TransitionRegion:
    corridor1_id: int
    corridor2_id: int
    polygon: Polygon
    representative_points: List[Tuple[float, float]]


class PlannerNode(Node):
    def __init__(self):
        super().__init__('corridor_planner')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('v_max', 0.5),
                ('omega_max', 2.0),
                ('robot_width', 0.34),
                ('robot_length', 0.237),
                ('robot_wheelbase', 0.25),
                ('robot_delta_max', 0.5),
                ('model_type', 'bicycle'),
                ('sampling_dt', 0.100),
                ('map_frame', 'map'),
                # ---- new tuning knobs ----
                ('corridor_change_penalty', 0.05),  # small tie-break, keep << shortest meaningful length gap
                ('redundancy_tol', 0.10),           # min length a corridor must save to be kept (~robot scale)
                ('max_reroutes', 3),                # bounded safety net; ~never fires at small R
                ('terminal_heading_weight', 5.0),   # bias (m/rad) toward a start/goal corridor whose axis matches the pose heading
                ('terminal_heading_mode', 'snap'),  # 'strict' = exact pose or no solution; 'snap' = deliver nearest reachable heading and report
                ('allow_reverse_parking', False),    # False = only enter a narrow terminal driving forward onto the pose (more predictable, forces U-turns)
                ('debug_context', True),            # dump graph_context.json for offline analysis
            ]
        )

        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('omega_max').value
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_wheelbase = self.get_parameter('robot_wheelbase').value
        self.delta_max = self.get_parameter('robot_delta_max').value
        self.model_type = self.get_parameter('model_type').value.lower()
        self.dt = self.get_parameter('sampling_dt').value
        self.map_frame = self.get_parameter('map_frame').value
        self.corridor_change_penalty = self.get_parameter('corridor_change_penalty').value
        self.redundancy_tol = self.get_parameter('redundancy_tol').value
        self.max_reroutes = int(self.get_parameter('max_reroutes').value)
        self.terminal_heading_weight = self.get_parameter('terminal_heading_weight').value
        self.terminal_heading_mode = self.get_parameter('terminal_heading_mode').value
        self.allow_reverse_parking = bool(self.get_parameter('allow_reverse_parking').value)
        self.debug_context = bool(self.get_parameter('debug_context').value)

        # Kept only for clamp_pose_to_corridor (unchanged behaviour).
        self.safety_margin = max(self.robot_width, self.robot_length) / 2.0 + 0.02
        self.clamping = False

        self.graph = None
        self.G = None
        self.transition_graph = None
        self.corridor_polygons = {}
        self.transition_regions = {}
        self.nodes_by_corridor = {}

        self.initial_point = (0.0, 0.0)
        self.target_point = None
        self.current_state = None
        self.initial_angle = 0.0
        self.target_angle = None
        self.prev_corridor_marker_count = 0

        # ---------------------------------------------------------
        # VEHICLE & PLANNER INIT
        # ---------------------------------------------------------
        self.vehicle = self._create_vehicle_model()

        # Footprint disk radius r and turning radius R, as functions of the vehicle
        # (never hard-coded to a specific platform). The incoming corridors are the
        # FULL free-space rectangles: robot_clearance in the generator only crops
        # corridors/overlaps too small for the robot, it does not deflate them.
        if self.model_type == 'bicycle':
            self.r = self.robot_width / 2.0
            self.turning_radius = getattr(self.vehicle, 'max_radius', None)
            if self.turning_radius is None:
                try:
                    self.turning_radius = abs(self.robot_wheelbase / math.tan(self.delta_max))
                except Exception:
                    self.turning_radius = None
        else:
            # Unicycle: enclosing disk, turns on the spot, so no arc/triplet rules.
            self.r = max(self.robot_width, self.robot_length) / 2.0
            self.turning_radius = None

        self.get_logger().info(
            f"Planner initialized: Model={self.model_type}, r={self.r:.3f}, "
            f"R={self.turning_radius if self.turning_radius is None else round(self.turning_radius, 3)}, "
            f"terminal_heading_mode={self.terminal_heading_mode}, "
            f"allow_reverse_parking={self.allow_reverse_parking}"
        )

        self.plan_motion_node = PlanMotion(self.get_logger())
        if self.vehicle:
            self.plan_motion_node.initialize_planner(self.vehicle, sampling_time=self.dt)

        # ---------------------------------------------------------
        # ROS INTERFACES
        # ---------------------------------------------------------
        self.cli = self.create_client(GetGraph, '/get_graph')

        self.path_marker_pub = self.create_publisher(Marker, 'path_marker', 1)
        self.point_marker_pub = self.create_publisher(Marker, 'point_markers', 2)
        self.corridor_marker_pub = self.create_publisher(MarkerArray, 'corridor_markers', 1)
        self.transition_marker_pub = self.create_publisher(MarkerArray, 'transition_markers', 1)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, 'waypoint_markers', 1)
        self.planned_path_publisher = self.create_publisher(Path, '/plan', 10)

        self.create_subscription(Empty, '/floorplan_updated', self.floorplan_updated_callback, 10)
        self.create_subscription(Point, '/initial_point', self.initial_point_callback, 10)
        self.create_subscription(Point, '/target_point', self.target_point_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        self.create_subscription(Float64MultiArray, '/state', self.state_listener_callback, 10)

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_graph service...')
        self.request_graph()

    def _create_vehicle_model(self):
        try:
            if self.model_type == 'unicycle':
                return arena.Unicycle(
                    state=[0.0] * 3, width=self.robot_width, length=self.robot_length,
                    v_max=self.v_max, v_min=-self.v_max,
                    omega_max=self.w_max, omega_min=-self.w_max
                )
            elif self.model_type == 'bicycle':
                return arena.Bicycle(
                    state=[0.0] * 3, width=self.robot_width, length=self.robot_length,
                    wheelbase=self.robot_wheelbase, v_max=self.v_max, v_min=-self.v_max,
                    delta_max=self.delta_max, delta_min=-self.delta_max
                )
            else:
                self.get_logger().fatal(f"Unknown model_type: {self.model_type}")
                return None
        except Exception as e:
            self.get_logger().fatal(f"Failed to create vehicle: {e}")
            return None

    # ---------------------------------------------------------
    # GRAPH LIFECYCLE
    # ---------------------------------------------------------
    def request_graph(self):
        future = self.cli.call_async(GetGraph.Request())
        future.add_done_callback(self.on_graph_response)

    def floorplan_updated_callback(self, _):
        self.request_graph()

    def on_graph_response(self, future):
        try:
            response = future.result()
            self.graph = response.graph
            self.G = self.build_graph_from_msg(response.graph)
            # STAGE 1: vehicle-specific edge admissibility on the raw rectangle graph.
            self.apply_edge_admissibility()
            # STAGE 2: transition regions + graph built only over admissible edges.
            self.compute_transition_regions()
            self.build_transition_graph()
            self.get_logger().info(f'Graph built: {len(self.G.nodes)} nodes, '
                                   f'{self.G.number_of_edges()} admissible edges.')
            self.try_plan_path()
        except Exception as e:
            self.get_logger().error(f'Failed to get graph: {e}')

    def state_listener_callback(self, msg):
        self.current_state = msg.data
        self.initial_point = (self.current_state[0], self.current_state[1])
        self.initial_angle = self.current_state[2]

    def initial_point_callback(self, msg):
        self.initial_point = (msg.x, msg.y)
        self.initial_angle = 0.0
        self.get_logger().info("Initial point set.")

    def target_point_callback(self, msg):
        self.target_point = (msg.x, msg.y)
        self.target_angle = 0.0
        self.try_plan_path()

    def goal_pose_callback(self, msg):
        self.target_point = (msg.pose.position.x, msg.pose.position.y)
        self.target_angle = tf_transformations.euler_from_quaternion([
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ])[2]
        self.get_logger().info(f"Target set from RViz: {self.target_point}")
        self.try_plan_path()

    def initial_pose_callback(self, msg):
        self.initial_point = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.initial_angle = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        ])[2]
        self.get_logger().info(f"Start set from RViz: {self.initial_point}")

    def create_corridor_polygon(self, cx, cy, width, height, yaw):
        half_w, half_h = width / 2, height / 2
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        poly = Polygon(corners)
        poly = rotate(poly, yaw, origin=(0, 0), use_radians=True)
        return translate(poly, xoff=cx, yoff=cy)

    def build_graph_from_msg(self, graph_msg):
        G = nx.Graph()
        valid_nodes = set()
        for corridor in graph_msg.nodes:
            node_id = corridor.id
            try:
                poly = self.create_corridor_polygon(corridor.center_x, corridor.center_y, corridor.width, corridor.height, corridor.yaw)
                self.corridor_polygons[node_id] = poly
                G.add_node(node_id, pos=(corridor.center_x, corridor.center_y), width=corridor.width, height=corridor.height, yaw=corridor.yaw, polygon=poly)
                valid_nodes.add(node_id)
            except Exception as e:
                self.get_logger().error(f"Failed to create corridor {node_id}: {e}")
        for edge in graph_msg.edges:
            if edge.from_corridor in valid_nodes and edge.to_corridor in valid_nodes:
                G.add_edge(edge.from_corridor, edge.to_corridor)
        # Spatial index over corridor polygons: lets us ask "is this point free space"
        # (covered by some corridor) cheaply, which the transition-corner filter needs.
        self._poly_list = list(self.corridor_polygons.values())
        self._corridor_tree = STRtree(self._poly_list) if self._poly_list else None
        return G

    # =========================================================
    # STAGE 1 - EDGE ADMISSIBILITY (vehicle-specific)
    # =========================================================
    def apply_edge_admissibility(self):
        """Prune corridor-graph edges the vehicle cannot actually use.

        Both modes: the shared overlap (gateway) must admit the footprint disk,
        i.e. its smaller dimension >= 2r. Narrow corridors self-isolate this way
        because one overlap dimension is bounded by their short width.

        Bicycle only: an orthogonal pair implies a 90-degree turn, so the corner
        arc must also fit (Sonia's pairwise inequality). Parallel pairs are a
        straight pass and only need the gateway. A too-tight turn is dropped here,
        which is exactly why a corridor whose direct gateway is too thin (Claim B's
        c35 case) survives as a needed connector rather than being pruned later.
        """
        for a, b in list(self.G.edges()):
            if not self._edge_feasible(a, b):
                self.G.remove_edge(a, b)

    def _edge_feasible(self, a, b) -> bool:
        gw = self._overlap_gateway(a, b)
        if gw is None or gw < 2.0 * self.r - 1e-6:
            return False
        if self.model_type == 'bicycle' and self._corridor_axis(a) != self._corridor_axis(b):
            if not self._pairwise_arc_feasible(self._short_width(a), self._short_width(b)):
                return False
        return True

    def _pairwise_arc_feasible(self, w1, w2) -> bool:
        """Sonia's corner-arc feasibility: max(0,R+r-w1)^2 + max(0,R+r-w2)^2 <= (R-r)^2,
        with w1,w2 >= 2r. Wide channels pass trivially; the narrow rule (w=2r forces
        the other >= R+r) falls out of the inequality."""
        R, r = self.turning_radius, self.r
        if R is None or R <= r:
            return (w1 >= 2.0 * r - 1e-6) and (w2 >= 2.0 * r - 1e-6)
        if w1 < 2.0 * r - 1e-6 or w2 < 2.0 * r - 1e-6:
            return False
        a = max(0.0, R + r - w1)
        b = max(0.0, R + r - w2)
        return a * a + b * b <= (R - r) ** 2 + 1e-9

    # ---- axis-aligned geometry helpers (yaw-independent, read from polygons) ----
    def _bbox_dims(self, cid) -> Tuple[float, float]:
        minx, miny, maxx, maxy = self.corridor_polygons[cid].bounds
        return (maxx - minx, maxy - miny)

    def _corridor_axis(self, cid) -> str:
        dx, dy = self._bbox_dims(cid)
        return 'H' if dx >= dy else 'V'

    def _short_width(self, cid) -> float:
        dx, dy = self._bbox_dims(cid)
        return min(dx, dy)

    def _overlap_geom(self, a, b):
        pa, pb = self.corridor_polygons.get(a), self.corridor_polygons.get(b)
        if pa is None or pb is None or not pa.intersects(pb):
            return None
        inter = pa.intersection(pb)
        if inter.is_empty:
            return None
        return inter

    def _overlap_gateway(self, a, b) -> Optional[float]:
        inter = self._overlap_geom(a, b)
        if inter is None:
            return None
        minx, miny, maxx, maxy = inter.bounds
        return min(maxx - minx, maxy - miny)

    def _overlap_centroid(self, a, b) -> Optional[Tuple[float, float]]:
        inter = self._overlap_geom(a, b)
        if inter is None:
            return None
        c = inter.centroid
        return (c.x, c.y)

    # =========================================================
    # STAGE 2 - TRANSITION REGIONS AND TRANSITION GRAPH
    # =========================================================
    def compute_transition_regions(self):
        self.transition_regions = {}
        self.get_logger().info(f"Computing transition regions for {self.G.number_of_edges()} admissible edges")
        for edge in self.G.edges():
            c1, c2 = edge
            poly1 = self.corridor_polygons.get(c1)
            poly2 = self.corridor_polygons.get(c2)
            if not poly1 or not poly2 or not poly1.intersects(poly2):
                continue

            intersection = poly1.intersection(poly2)
            if intersection.is_empty:
                continue

            if intersection.geom_type != 'Polygon':
                if hasattr(intersection, 'centroid'):
                    self.transition_regions[(c1, c2)] = TransitionRegion(c1, c2, None, [(intersection.centroid.x, intersection.centroid.y)])
                continue

            if intersection.area < 0.01:
                centroid = intersection.centroid
                self.transition_regions[(c1, c2)] = TransitionRegion(c1, c2, intersection, [(centroid.x, centroid.y)])
                continue

            points = self.generate_transition_points(intersection)
            self.transition_regions[(c1, c2)] = TransitionRegion(c1, c2, intersection, points)

    def _covered(self, x, y):
        """True if (x, y) is inside some corridor, i.e. free space in the cover."""
        if self._corridor_tree is None:
            return False
        p = ShapelyPoint(x, y)
        for idx in self._corridor_tree.query(p):
            if self._poly_list[idx].covers(p):
                return True
        return False

    def _is_convex_corner(self, x, y, eps):
        """A taut path bends only at convex obstacle corners, the reflex vertices of
        free space. Sampling the four diagonal neighbours of a point, such a corner has
        exactly one obstacle quadrant (three free). An overlap centroid has zero, a
        point flush against a flat wall has two, so this keeps only the real turning
        points and drops the rest."""
        obst = 0
        for dx in (-eps, eps):
            for dy in (-eps, eps):
                if not self._covered(x + dx, y + dy):
                    obst += 1
        return obst == 1

    def generate_transition_points(self, intersection):
        """Transition points for an overlap are only its convex-corner vertices, the
        ones a shortest path could actually bend at. This drops the centroid and every
        wall-flush corner, so a road-style crossing keeps its four corners, a corridor
        opening into a room keeps only its two door jambs, and the interior corners
        that the room already covers are discarded. On gmap_6 this cuts the transition
        node count by about 63 percent, which is the dominant cost in graph build and
        search. Connectivity is preserved because corridors are convex, so any two
        nodes sharing a corridor stay mutually visible, and each overlap keeps at least
        one node via the centroid fallback below."""
        x0, y0, x1, y1 = intersection.bounds
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            c = intersection.centroid
            return [(c.x, c.y)]
        eps = max(0.005, min(0.03, 0.25 * min(w, h)))
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        kept = [c for c in corners if self._is_convex_corner(c[0], c[1], eps)]
        if not kept:  # open overlap with no convex corner: keep centroid for connectivity
            c = intersection.centroid
            kept = [(c.x, c.y)]

        seen, unique = set(), []
        for pt in kept:
            pr = (round(pt[0], 3), round(pt[1], 3))
            if pr not in seen:
                seen.add(pr)
                unique.append(pt)
        return unique

    def build_transition_graph(self):
        self.transition_graph = nx.DiGraph()
        self.transition_graph.add_node('start')
        self.transition_graph.add_node('end')

        points_by_corridor = {}
        nc = 0
        for (c1, c2), region in self.transition_regions.items():
            for pt in region.representative_points:
                nid = f"t_{nc}"; nc += 1
                self.transition_graph.add_node(nid, point=pt, corridors={c1, c2}, region=(c1, c2))
                for c in [c1, c2]:
                    points_by_corridor.setdefault(c, []).append(nid)

        for cid, data in self.G.nodes(data=True):
            nid = f"c_{cid}"
            self.transition_graph.add_node(nid, point=data['pos'], corridors={cid}, region=None)
            points_by_corridor.setdefault(cid, []).append(nid)

        for cid, nodes in points_by_corridor.items():
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    pt1 = self.transition_graph.nodes[n1]['point']
                    pt2 = self.transition_graph.nodes[n2]['point']
                    if self.is_path_in_corridor(pt1, pt2, cid):
                        dist = math.dist(pt1, pt2)
                        if self.transition_graph.nodes[n1]['region'] and self.transition_graph.nodes[n2]['region']:
                            dist += self.corridor_change_penalty
                        self.transition_graph.add_edge(n1, n2, weight=dist)
                        self.transition_graph.add_edge(n2, n1, weight=dist)

        self.nodes_by_corridor = points_by_corridor
        self.visualize_transition_points()
        if self.debug_context:
            self._save_graph_context()

    def is_path_in_corridor(self, pt1, pt2, corridor_id):
        line = LineString([pt1, pt2])
        poly = self.corridor_polygons[corridor_id]
        if poly.contains(line):
            return True
        try:
            return poly.intersection(line).length >= 0.999 * line.length
        except Exception:
            return False

    # =========================================================
    # STAGE 0 - CONTAINMENT WITH FOOTPRINT (start/goal selection)
    # =========================================================
    def find_corridors_containing_point(self, point):
        """Corridors whose FOOTPRINT-deflated polygon contains the point, i.e. where
        a disk of radius r centred at the point fits. Deflating by r is what stops
        the planner from starting inside a small corridor nested in a larger one when
        the click sits too close to the small corridor's wall (issue 2)."""
        p = ShapelyPoint(point[0], point[1])
        out = []
        for cid, poly in self.corridor_polygons.items():
            safe = poly.buffer(-self.r)
            if not safe.is_empty and safe.contains(p):
                out.append(cid)
        return out

    def _clearance(self, cid, point) -> float:
        poly = self.corridor_polygons[cid]
        return poly.exterior.distance(ShapelyPoint(point[0], point[1]))

    def _best_corridor(self, candidates, point):
        """Among containing corridors pick the one with the most footprint clearance,
        so a click in a big room is not attributed to a thin nested corridor."""
        return max(candidates, key=lambda cid: self._clearance(cid, point))

    def _nearest_corridor(self, point):
        p = ShapelyPoint(point[0], point[1])
        return min(self.corridor_polygons, key=lambda cid: self.corridor_polygons[cid].distance(p))

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _terminal_entry_ok(self, T, nid, heading, point):
        """Can the vehicle hold a pose with `heading` at `point` in narrow terminal
        corridor T, entering or leaving through transition node `nid`?

        A room, or a unicycle, holds any heading: passes. In a narrow corridor the
        heading is pinned to the axis. An entry from a PARALLEL corridor is a straight
        in or reverse move, always fine, and already vouched for by the Stage 1 two and
        three corridor checks (HV, VH, VHV, HVH) plus the pose lying inside the shrunk
        corridor. An entry from an ORTHOGONAL corridor must arc from that corridor's
        axis onto T's. If the target lies on the arc's own side (front parking) the arc
        ends on the already validated pose, so there is nothing to check. If it lies on
        the far side, the arc first drives AWAY to establish the heading and then
        reverses back, and that outward tangent must clear T's far wall.

        The entry corridor's EDGE along T's axis is used, not its centreline, since the
        vehicle may sit anywhere across N with r clearance and start the arc from the
        most favourable position:

            heading toward -axis (f = -1):  N_far  - r - R >= T_near_wall + r
            heading toward +axis (f = +1):  N_near + r + R <= T_far_wall  - r

        Front parking satisfies this automatically. The test only bites a reverse
        maneuver whose outward arc would cross the far wall, e.g. a vertical corridor
        hugging the left end of a narrow horizontal terminal with a pi goal heading.
        When it prunes the shortest entry, Dijkstra takes the next orthogonal corridor
        that clears, producing the longer V->H or H->V->H route around the obstacle.

        With allow_reverse_parking False the same inequality is used but the far WALL is
        replaced by the POSE itself, which demands the arc land on or beyond the pose so
        the vehicle only ever drives forward onto it. The arc leaves N's axis and meets
        T's centreline a distance R along the heading, so the reachable tangents span
        [N_near + r + R, N_far - r + R] (mirrored for -axis); requiring the pose to lie
        within that span is exactly the forward condition. Since the pose already lies
        inside the shrunk corridor, this is strictly stronger than the wall test, and it
        forces the U-turn around the obstacle wherever a forward entry is impossible."""
        if heading is None or self.turning_radius is None:
            return True
        wT, hT = self.G.nodes[T]['width'], self.G.nodes[T]['height']
        if min(wT, hT) >= 2.0 * (self.turning_radius + self.r):
            return True  # room: any heading reachable
        cs = self.transition_graph.nodes[nid].get('corridors') or set()
        others = [c for c in cs if c != T]
        if not others:
            return True
        N = others[0]
        wN, hN = self.G.nodes[N]['width'], self.G.nodes[N]['height']
        if (wT >= hT) == (wN >= hN):
            return True  # aligned entry: straight in / reverse, vouched by Stage 1
        R, r = self.turning_radius, self.r
        cx, cy = self.G.nodes[T]['pos']
        nx, ny = self.G.nodes[N]['pos']
        if wT >= hT:                       # horizontal terminal, axis x
            f = 1 if math.cos(heading) >= 0 else -1
            Tlo, Thi = cx - wT / 2.0, cx + wT / 2.0
            Nlo, Nhi = nx - wN / 2.0, nx + wN / 2.0
        else:                              # vertical terminal, axis y
            f = 1 if math.sin(heading) >= 0 else -1
            Tlo, Thi = cy - hT / 2.0, cy + hT / 2.0
            Nlo, Nhi = ny - hN / 2.0, ny + hN / 2.0
        if self.allow_reverse_parking:
            lim_pos, lim_neg = Thi - r, Tlo + r        # arc must clear T's far wall
        else:
            g = point[0] if wT >= hT else point[1]     # arc must reach the pose itself
            lim_pos = lim_neg = g
        if f > 0:
            return Nlo + r + R <= lim_pos + 1e-9
        return Nhi - r - R >= lim_neg - 1e-9

    def _terminal_bias(self, cid, heading):
        """Penalty (in metres) for entering or leaving the plan through corridor `cid`
        at the commanded `heading`. The start and goal attach to every containing
        corridor, and pure length would pick whichever is shortest regardless of whether
        the pose can be held there. This bias makes the terminal choice heading-aware.

        A unicycle turns in place, so any heading is reachable and the bias is zero. For
        the bicycle, what matters is the absolute short dimension: at least 2(R + r) and
        the corridor holds a full turning circle, so it is a room and any heading is
        reachable, bias zero. A narrower corridor is traversed along its long axis, so
        the reachable heading is that axis in either direction and the bias grows with
        the gap between the commanded heading and that axis. This is what steers a
        horizontal goal heading into a horizontal terminal corridor rather than a
        vertical one."""
        if heading is None or self.turning_radius is None:
            return 0.0
        w = self.G.nodes[cid]['width']
        h = self.G.nodes[cid]['height']
        if min(w, h) >= 2.0 * (self.turning_radius + self.r):
            return 0.0  # room: any heading reachable
        axis = 0.0 if w >= h else math.pi / 2.0
        gap = min(abs(self._wrap(heading - axis)), abs(self._wrap(heading - axis - math.pi)))
        return self.terminal_heading_weight * gap

    # =========================================================
    # PLANNING ENTRY POINT
    # =========================================================
    def _snap_target(self, goal_cs, contained):
        """Resolve the commanded target pose against the corridors, per
        terminal_heading_mode.

        Two independent things can be unreachable, and they are snapped in the order
        the geometry forces: POSITION first, since a point outside every corridor (or
        inside one only if the footprint is ignored) has no corridor to reason about,
        so it is pulled to the nearest valid pose inside the nearest corridor. HEADING
        second, since only once the terminal corridor is known can we tell whether the
        heading is holdable there: a room holds any heading, but a narrow corridor
        (short side below 2(R + r), in the limit ~2r) pins the vehicle to its axis, so
        the heading must snap to 0, pi, or +-pi/2, whichever of the axis directions is
        closest to the commanded one.

        strict: nothing is snapped. An unreachable pose is reported and the query is
        refused, so the caller learns the exact pose is impossible rather than being
        silently given a different one.
        snap: position and heading are corrected as above, and every correction is
        reported. Position is preserved wherever possible, the heading yields."""
        strict = (self.terminal_heading_mode == 'strict')

        if not contained:                      # position not inside any corridor
            cid = goal_cs[0]
            if strict:
                self.get_logger().warn(
                    "Target position is not inside any corridor with footprint clearance; "
                    "refusing (terminal_heading_mode=strict).")
                return False
            clamped = self.clamp_pose_to_corridor(self.target_point, cid)
            self.get_logger().warn(
                f"Target position {self.target_point} outside all corridors; snapped to "
                f"({clamped[0]:.3f}, {clamped[1]:.3f}) inside corridor {cid}.")
            self.target_point = (clamped[0], clamped[1])

        if self.target_angle is None or self.turning_radius is None:
            return True

        # Heading: only a narrow terminal corridor constrains it. If ANY containing
        # corridor is a room, or holds the heading on its axis, the heading stands.
        cands = self.find_corridors_containing_point(self.target_point) or goal_cs
        best_gap, best_axis = None, None
        for cid in cands:
            w, h = self.G.nodes[cid]['width'], self.G.nodes[cid]['height']
            if min(w, h) >= 2.0 * (self.turning_radius + self.r):
                return True                    # a room contains it: any heading is fine
            axis = 0.0 if w >= h else math.pi / 2.0
            for cand in (axis, self._wrap(axis + math.pi)):
                gap = abs(self._wrap(self.target_angle - cand))
                if best_gap is None or gap < best_gap:
                    best_gap, best_axis = gap, cand
        if best_gap is None or best_gap <= 0.10:
            return True                        # heading already lies on a reachable axis

        if strict:
            self.get_logger().warn(
                f"Goal heading {math.degrees(self.target_angle):.0f} deg cannot be held in a "
                f"narrow terminal corridor (nearest reachable {math.degrees(best_axis):.0f} deg); "
                f"refusing (terminal_heading_mode=strict).")
            return False
        self.get_logger().warn(
            f"Goal heading {math.degrees(self.target_angle):.0f} deg not holdable in narrow "
            f"terminal corridor; snapped to {math.degrees(best_axis):.0f} deg "
            f"(corrected by {math.degrees(best_gap):.0f} deg).")
        self.target_angle = best_axis
        return True

    def try_plan_path(self):
        if self.transition_graph is None or self.target_point is None:
            return

        start_cs = self.find_corridors_containing_point(self.initial_point)
        goal_cs = self.find_corridors_containing_point(self.target_point)
        if not start_cs:
            start_cs = [self._nearest_corridor(self.initial_point)]
        contained = bool(goal_cs)
        if not goal_cs:
            goal_cs = [self._nearest_corridor(self.target_point)]

        # Resolve the target pose (position, then heading) before any search.
        if not self._snap_target(goal_cs, contained):
            return
        goal_cs = self.find_corridors_containing_point(self.target_point) or goal_cs

        # Single-corridor direct case.
        common = set(start_cs) & set(goal_cs)
        if common:
            cid = self._best_corridor(common, self.initial_point)
            if self.is_path_in_corridor(self.initial_point, self.target_point, cid):
                waypoints = [self.initial_point, self.target_point]
                self.execute_motion_planning([cid], waypoints, None, one_corridor=True)
                self.publish_viz(waypoints, [cid])
                return

        t_start = time.time()
        result = self._plan_sequence(start_cs, goal_cs)
        self.get_logger().info(f"Corridor search + prune + validate: {(time.time()-t_start)*1000:.2f} ms")

        if result is None:
            self.get_logger().warn("No feasible corridor path found.")
            return

        sequence, tilts, waypoints = result
        self.execute_motion_planning(sequence, waypoints, None, one_corridor=False, precomputed_tilts=tilts)
        self.publish_viz(waypoints, sequence)

    def _plan_sequence(self, start_cs, goal_cs):
        """Search -> prune -> tilt -> validate, with a bounded reroute net.

        The reroute only fires on a genuinely infeasible triple (0 < centerline < 2R),
        which is empirically ~0% at the current R. It blocks the offending turn's
        transition region and re-searches. Blocking a whole region is a coarse forbid
        (it cannot say 'this pair is fine from a different predecessor'); the exact
        forbid needs the second-order graph, which is the intended large-R upgrade.
        """
        blocked: Set[frozenset] = set()
        for attempt in range(self.max_reroutes + 1):
            path = self._run_search(start_cs, goal_cs, blocked)
            if path is None:
                return None

            # raw corridor sequence + the real entry/exit crossing point per corridor.
            raw_seq, crossings = self._extract(path, start_cs, goal_cs)
            if not raw_seq:
                return None

            # STAGE 4a: drop only corridors that save negligible length vs a direct
            # neighbour-to-neighbour connection (clips, not shortcut-bearing rooms).
            kept = self.prune_redundant_corridors(raw_seq, crossings)
            sequence = [raw_seq[i] for i in kept]
            ee = [crossings[i] for i in kept]  # each retained corridor keeps its own crossings

            # STAGE 5: tilts from corridor shape (long axis) plus crossing sign.
            tilts = self.compute_corridor_tilts(sequence, ee)
            # STAGE 4b: triplet feasibility (bicycle only).
            bad = self.validate_sequence(sequence, tilts)

            if not bad:
                waypoints = self._waypoints_from_crossings(ee)
                self._log_sequence(path, sequence, tilts)
                return sequence, tilts, waypoints

            k = bad[0]  # middle corridor index of the first infeasible triple
            pair = frozenset((sequence[k], sequence[k + 1])) if k + 1 < len(sequence) \
                else frozenset((sequence[k - 1], sequence[k]))
            blocked.add(pair)
            self.get_logger().warn(
                f"Infeasible maneuver at corridor {sequence[k]} "
                f"(centerline < 2R); blocking {tuple(pair)} and rerouting "
                f"(attempt {attempt + 1}/{self.max_reroutes})."
            )
        self.get_logger().warn("Reroute budget exhausted; no feasible sequence.")
        return None

    def _blocked_terminal_nodes(self, cs, heading, point):
        """Transition nodes through which a terminal corridor cannot be entered or left
        while holding `point`/`heading`.

        Gating the ATTACHMENT edge alone is not enough, for two reasons, and both bite
        in practice. The transition graph also carries one CORRIDOR-CENTRE node per
        corridor, which belongs to a single corridor and therefore has no entry to test,
        so the pose attaches to it and the gate is skipped: on gmap_6 every t_* entry to
        c53 was correctly refused for a pi heading, yet the goal still attached to c_53
        and the search walked in through the forbidden c45. And since any two nodes
        inside one convex corridor are mutually visible, a path could otherwise enter
        through a forbidden node and slide across to a permitted one.

        Blocking the nodes closes both. The centre node stays reachable only through the
        corridor's own admissible entries, which is exactly the intent. This is safe
        because the pose lies in this corridor, so a shortest path enters it once and
        stops."""
        out = set()
        for cid in cs:
            for nid in self.nodes_by_corridor.get(cid, []):
                if not self._terminal_entry_ok(cid, nid, heading, point):
                    out.add(nid)
        return out

    def _run_search(self, start_cs, goal_cs, blocked: Set[frozenset]):
        G = self.transition_graph
        # Attach virtual start/end to every candidate corridor's transition nodes, with a
        # per-corridor heading bias so the search prefers a terminal corridor whose axis
        # matches the commanded pose heading, while keeping every corridor as a fallback.
        for cid in start_cs:
            bias = self._terminal_bias(cid, getattr(self, 'initial_angle', None))
            for nid in self.nodes_by_corridor.get(cid, []):
                pt = G.nodes[nid]['point']
                if self.is_path_in_corridor(self.initial_point, pt, cid) \
                        and self._terminal_entry_ok(cid, nid, getattr(self, 'initial_angle', None),
                                                    self.initial_point):
                    G.add_edge('start', nid, weight=math.dist(self.initial_point, pt) + bias)
        for cid in goal_cs:
            bias = self._terminal_bias(cid, getattr(self, 'target_angle', None))
            for nid in self.nodes_by_corridor.get(cid, []):
                pt = G.nodes[nid]['point']
                if self.is_path_in_corridor(pt, self.target_point, cid) \
                        and self._terminal_entry_ok(cid, nid, getattr(self, 'target_angle', None),
                                                    self.target_point):
                    G.add_edge(nid, 'end', weight=math.dist(pt, self.target_point) + bias)

        blocked_nodes = [nid for nid, data in G.nodes(data=True)
                         if data.get('region') and frozenset(data['region']) in blocked]
        # Forbid entering/leaving a narrow terminal corridor through a node whose
        # maneuver cannot hold the pose. Node-level, not edge-level: see
        # _blocked_terminal_nodes for why the attachment gate alone leaks.
        blocked_nodes += list(self._blocked_terminal_nodes(
            goal_cs, getattr(self, 'target_angle', None), self.target_point))
        blocked_nodes += list(self._blocked_terminal_nodes(
            start_cs, getattr(self, 'initial_angle', None), self.initial_point))
        path = None
        try:
            view = nx.restricted_view(G, blocked_nodes, [])
            path = nx.shortest_path(view, 'start', 'end', weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = None
        finally:
            G.remove_edges_from(list(G.out_edges('start')))
            G.remove_edges_from(list(G.in_edges('end')))
        return path

    def _extract(self, path, start_cs, goal_cs):
        """Walk the transition-node path, read off the shared corridor per hop, and
        record where the path actually enters and leaves each corridor. Consecutive
        segments in the same corridor are merged so a corridor's entry is its first
        crossing and its exit is its last. Returns (raw_sequence, crossings) where
        crossings[i] = (entry_point, exit_point) for raw_sequence[i]."""
        pts, csets = [], []
        for n in path:
            if n == 'start':
                pts.append(self.initial_point); csets.append(set(start_cs))
            elif n == 'end':
                pts.append(self.target_point); csets.append(set(goal_cs))
            else:
                d = self.transition_graph.nodes[n]
                pts.append(tuple(d['point'])); csets.append(set(d.get('corridors', set())))

        raw, crossings = [], []
        prev = None
        for i in range(len(pts) - 1):
            sh = csets[i] & csets[i + 1]
            c = prev if (sh and prev in sh) else (min(sh) if sh else prev)
            if c is None:
                continue
            if raw and raw[-1] == c:
                crossings[-1] = (crossings[-1][0], pts[i + 1])  # extend exit
            else:
                raw.append(c)
                crossings.append((pts[i], pts[i + 1]))
            prev = c
        return raw, crossings

    # =========================================================
    # STAGE 4a - REDUNDANCY PRUNE
    # =========================================================
    def prune_redundant_corridors(self, seq, crossings):
        """Drop a middle corridor only when routing through it saves negligible path
        length versus connecting its neighbours directly through their shared overlap.
        This is the distinction between a genuine shortcut and a clip.

        A room in the middle of the sequence is usually what lets the AP cut a diagonal
        across it; removing it forces an L and a LONGER trajectory. So the corridor tube
        exists to give the trajectory room, and more corridors mean shorter paths, not
        longer. Shortest corridor count is never the goal, shortest trajectory is. The
        earlier "do the neighbours touch" and "does the crossing land in a neighbour"
        tests were both wrong: on map_14 the clipped corridor c9 fills a tiny corner
        that no single corridor nor even the neighbour union covers, so no exact
        geometric containment can catch it, yet it saves nothing and should go, while
        the diagonal room c8 has overlapping neighbours too but saves a lot and must
        stay. Only a length comparison separates them.

        For middle M with predecessor P and successor S, let a be P's entry crossing,
        e and x be M's entry/exit crossings, and b be S's exit crossing. L_with is the
        current route a->e->x->b through M. L_without is the shortest a->b that stays in
        P and S, i.e. the straight segment if it already passes through the P-S overlap,
        otherwise a bend at the overlap. M is redundant when L_without - L_with is below
        redundancy_tol, a length on the order of the robot's own size (a shortcut smaller
        than the vehicle is not worth a corridor). On map_14 this drops only c9 (saves
        0.03 m); c10's descent (0.30 m) and c8's diagonal (1.92 m) stay, as do the doors
        c14, c1, c3 whose neighbours do not connect directly.

        We only drop when the P-S edge is admissible, so the resulting adjacency is
        footprint-feasible. Candidates are removed smallest-saving first, each guarded
        by a live connectivity check so a removal never breaks the chain. Returns kept
        indices; a retained corridor keeps its own crossings, so tilts are unaffected.
        """
        N = len(seq)
        if N < 3:
            return list(range(N))

        candidates = []
        for k in range(1, N - 1):
            P, S = seq[k - 1], seq[k + 1]
            if not self.G.has_edge(P, S):
                continue
            a = crossings[k - 1][0]
            e_mid, x_mid = crossings[k]
            b = crossings[k + 1][1]
            saving = self._shortcut_saving(P, S, a, e_mid, x_mid, b)
            if saving < self.redundancy_tol:
                candidates.append((saving, k))

        candidates.sort(key=lambda t: t[0])
        kept = list(range(N))
        for _, k in candidates:
            if k not in kept:
                continue
            j = kept.index(k)
            if j == 0 or j == len(kept) - 1:
                continue
            if self.G.has_edge(seq[kept[j - 1]], seq[kept[j + 1]]):
                kept.pop(j)
        return kept

    def _shortcut_saving(self, P, S, a, e_mid, x_mid, b):
        """Length saved by keeping the middle corridor: (shortest a->b through P and S
        only) minus (current a->e->x->b route through the middle). Small means the
        middle is a clip; large means it carries a real diagonal."""
        d = math.dist
        L_with = d(a, e_mid) + d(e_mid, x_mid) + d(x_mid, b)
        overlap = self.corridor_polygons[P].intersection(self.corridor_polygons[S])
        if overlap.is_empty:
            return float('inf')
        seg = LineString([a, b])
        if seg.intersects(overlap):
            L_without = d(a, b)
        else:
            g = nearest_points(overlap, seg)[0]
            L_without = d(a, (g.x, g.y)) + d((g.x, g.y), b)
        return L_without - L_with

    def _waypoints_from_crossings(self, ee):
        """Via-points for markers only (the AP consumes the corridor list). Uses the
        real exit crossing of each corridor, which coincides with the next entry."""
        wpts = [self.initial_point]
        for ent, ext in ee[:-1]:
            wpts.append(ext)
        wpts.append(self.target_point)
        return wpts

    # =========================================================
    # STAGE 5 - TILT ASSIGNMENT (from real crossing points; fixes map_14)
    # =========================================================
    def _gateway_span(self, a, b, k):
        """Projection of the a-b gateway onto axis k (0 = x, 1 = y) as (lo, hi)."""
        it = self.corridor_polygons[a].intersection(self.corridor_polygons[b])
        if it.is_empty:
            return None
        x0, y0, x1, y1 = it.bounds
        return (x0, x1) if k == 0 else (y0, y1)

    def _axis_sign(self, seq, ee, i, axis):
        """Signed progress of corridor i along `axis`.

        First ask whether the sign is ambiguous at all. The path enters corridor i
        somewhere in its entry gateway and leaves somewhere in its exit gateway, so if
        those two gateways do not overlap along the axis, EVERY possible pair of
        crossing points gives the same sign, no matter how short the traversal. Trust it
        and stop. This is the common case and it must not be second-guessed: gmap_6 c36
        is entered from c12 at y <= 5.41 and leaves into c22 at y >= 5.53, so it is
        unambiguously travelled upward, yet its own traversal is only 0.12 and the old
        magnitude test widened the window until it reached c45 and came back with c45's
        DOWNWARD direction, handing the AP a corridor pointing the wrong way.

        Only when the gateways overlap along the axis is the corridor crossed
        transversely and the corner picked able to flip the sign. Then fall back to the
        corridor's own traversal, and widen the window only if that traversal is shorter
        than the robot: map_14 c8 has nested gateways but travels -4.59, which is
        decisive, while gmap_6 c34 has overlapping gateways and travels only -0.08,
        which is noise and genuinely needs the wider window."""
        k = 0 if axis == 'H' else 1
        n = len(seq)
        ent = (ee[i][0][k],) * 2 if i == 0 else self._gateway_span(seq[i - 1], seq[i], k)
        ext = (ee[i][1][k],) * 2 if i == n - 1 else self._gateway_span(seq[i], seq[i + 1], k)
        if ent and ext:
            if ext[0] > ent[1] + 1e-9:
                return 1.0
            if ext[1] < ent[0] - 1e-9:
                return -1.0

        tol = 2.0 * self.r
        lo = hi = i
        while True:
            a, b = ee[lo][0], ee[hi][1]
            proj = b[k] - a[k]
            if abs(proj) >= tol or (lo == 0 and hi == n - 1):
                return proj
            lo, hi = max(0, lo - 1), min(n - 1, hi + 1)

    def compute_corridor_tilts(self, seq, ee):
        """Traversal direction through each corridor, snapped to an axis. The axis comes
        from the corridor's shape, the sign from the path's progress along that axis.

        An elongated corridor (aspect ratio >= 2) is ALWAYS traversed along its long
        axis, so the shape fixes the axis. This is the aspect-ratio rule from the paper
        and it is what gives the AP a turn at every transition: a long thin corridor the
        path merely clips would otherwise snap to the wrong axis, leaving two same-axis
        corridors in a row and no turn between them. It also matters where a corridor is
        crossed transversely to make a lateral step, as gmap_6 c34 does between c6 and
        c45, whose direct gateway is only 0.08 wide and therefore inadmissible: labelling
        c34 horizontal is exactly what makes the AP lay down the arc-straight-arc that
        performs the step. Only square-ish corridors (aspect < 2, a room) take their axis
        from the crossing, having no dominant one.

        The sign is then delegated to _axis_sign, which widens its window when the
        corridor's own traversal is too short to trust.

        The two TERMINAL corridors are special. In a narrow corridor the vehicle's
        heading is pinned to the axis, so the tilt is fully determined by the commanded
        pose and must not be inferred from the crossings at all: the pose IS the
        heading. Deriving it from displacement gets this wrong in two ways. With reverse
        parking the vehicle faces one way while moving the other, so displacement and
        heading are opposite by construction. And even in pure forward parking the final
        run is often shorter than 2r, which sends _axis_sign widening backwards into the
        previous corridor and inheriting ITS direction: on gmap_6 the last corridor c25
        is entered from c26 at x=1.04 with the goal at x=1.25, only 0.21 of travel, so
        the window widened into c1 (travelling -x) and returned pi for a goal heading of
        0. Rooms keep the inferred tilt, since a 3-point or reverse maneuver there can
        reach any heading and the traversal direction really is set by the path."""
        tilts = []
        n = len(seq)
        for i, cid in enumerate(seq):
            ent, ext = ee[i]
            dx, dy = ext[0] - ent[0], ext[1] - ent[1]
            w, h = self.G.nodes[cid]['width'], self.G.nodes[cid]['height']
            if w >= 2.0 * h:
                axis = 'H'
            elif h >= 2.0 * w:
                axis = 'V'
            else:                      # room: no dominant axis, take it from the crossing
                axis = 'H' if abs(dx) >= abs(dy) else 'V'
            proj = self._axis_sign(seq, ee, i, axis)
            if axis == 'H':
                tilts.append(0.0 if proj >= 0 else math.pi)
            else:
                tilts.append(math.pi / 2 if proj >= 0 else -math.pi / 2)

        # Terminal corridors: a narrow one pins the heading to its axis, so the pose
        # dictates the tilt. Start first, then goal, so a single-corridor sequence ends
        # on the goal heading.
        if n:
            pin = self._pinned_axis_tilt(seq[0], getattr(self, 'initial_angle', None))
            if pin is not None:
                tilts[0] = pin
            pin = self._pinned_axis_tilt(seq[-1], getattr(self, 'target_angle', None))
            if pin is not None:
                tilts[-1] = pin
        return tilts

    def _pinned_axis_tilt(self, cid, heading):
        """Tilt forced by a commanded pose in a NARROW corridor, where the vehicle's
        heading has to lie along the corridor axis: the axis direction closest to
        `heading`. Returns None for a room, or when there is no heading or no turning
        radius, leaving the tilt to be inferred from the path."""
        if heading is None or self.turning_radius is None:
            return None
        w, h = self.G.nodes[cid]['width'], self.G.nodes[cid]['height']
        if min(w, h) >= 2.0 * (self.turning_radius + self.r):
            return None                     # room: any heading reachable, path decides
        cands = (0.0, math.pi) if w >= h else (math.pi / 2.0, -math.pi / 2.0)
        return min(cands, key=lambda a: abs(self._wrap(heading - a)))

    def _fallback_tilts(self, seq):
        """Rougher tilt from consecutive-corridor overlap centroids. Only used if
        execute_motion_planning is ever called without precomputed tilts; the normal
        path uses compute_corridor_tilts on real crossings."""
        ee = []
        n = len(seq)
        for i, cid in enumerate(seq):
            ent = self.initial_point if i == 0 else self._overlap_centroid(seq[i - 1], cid)
            ext = self.target_point if i == n - 1 else self._overlap_centroid(cid, seq[i + 1])
            ee.append((ent if ent is not None else self.G.nodes[cid]['pos'],
                       ext if ext is not None else self.G.nodes[cid]['pos']))
        return self.compute_corridor_tilts(seq, ee)

    # =========================================================
    # STAGE 4b - TRIPLET FEASIBILITY (bicycle U/S maneuver)
    # =========================================================
    def _traversal_axis(self, tilt) -> str:
        return 'H' if abs(math.cos(tilt)) >= abs(math.sin(tilt)) else 'V'

    def _centerline_distance(self, a, m, b, ta, tm, tb) -> Optional[float]:
        """Distance between the outer corridors' parallel centerlines, classified by
        TRAVERSAL axis (so a corridor crossed short-ways is inverted, exactly as
        Sonia's invert_dimensions). Inverting keeps the centre and flips the axis
        label, so the distance reduces to |dx| or |dy| between the two centres."""
        A, M, B = self._traversal_axis(ta), self._traversal_axis(tm), self._traversal_axis(tb)
        xa, ya = self.G.nodes[a]['pos']
        xb, yb = self.G.nodes[b]['pos']
        dx, dy = abs(xa - xb), abs(ya - yb)
        case = A + M + B
        if case == 'HVH':
            return dy
        if case == 'VHV':
            return dx
        if case == 'HHV':
            return dx
        if case == 'VVH':
            return dy
        if case == 'HVV':
            return dy
        if case == 'VHH':
            return dx
        if case in ('HHH', 'VVV'):
            return max(dx, dy)
        return None

    def validate_sequence(self, seq, tilts):
        """Return the indices of infeasible middle corridors. A triple is feasible if
        the outer centerlines are at least 2R apart (room for two arcs). Runs after the
        prune, so collinear pass-throughs are already gone and only real turns remain."""
        bad = []
        if self.model_type != 'bicycle' or self.turning_radius is None:
            return bad
        need = 2.0 * self.turning_radius
        room = need + 2.0 * self.r  # arc diameter plus footprint
        for i in range(len(seq) - 2):
            mid = seq[i + 1]
            mw, mh = self.G.nodes[mid]['width'], self.G.nodes[mid]['height']
            # Sonia's outer-centerline rule assumes the middle corridor is a narrow
            # turning passage that pins the vehicle near the two centerlines. A room
            # wide enough to hold the whole arc does not, so the outer offset is
            # irrelevant and the turn is always feasible. map_14 c8 (13.8 x 5.0) is
            # exactly this: the check reads dx = 0.22 between the c1 and c3 centres and
            # would wrongly reject it, forcing a reroute that drops the diagonal room.
            if min(mw, mh) >= room:
                continue
            d = self._centerline_distance(seq[i], seq[i + 1], seq[i + 2],
                                          tilts[i], tilts[i + 1], tilts[i + 2])
            if d is not None and d < need - 1e-6:
                bad.append(i + 1)
        return bad

    # =========================================================
    # HAND-OFF TO ANALYTICAL PLANNER
    # =========================================================
    def execute_motion_planning(self, corridor_sequence, waypoints, waypoint_mapping,
                                one_corridor=False, precomputed_tilts=None):
        corridor_list = []
        if one_corridor:
            for cid in corridor_sequence:
                node = self.G.nodes[cid]
                corridor_list.append(Corridor.CorridorWorld(
                    width=node['height'], height=node['width'],
                    center=[node['pos'][0], node['pos'][1]], tilt=node['yaw']
                ))
        else:
            corridor_tilts = precomputed_tilts if precomputed_tilts is not None \
                else self._fallback_tilts(corridor_sequence)

            for i, cid in enumerate(corridor_sequence):
                node = self.G.nodes[cid]
                hw = np.array([node['width'], node['height']])
                h_final, w_final = np.abs(np.dot(self.R(corridor_tilts[i]), hw))
                corridor_list.append(Corridor.CorridorWorld(
                    width=w_final, height=h_final,
                    center=[node['pos'][0], node['pos'][1]],
                    tilt=node['yaw'] + corridor_tilts[i]
                ))
            self.get_logger().info(f"Corridor tilts: {corridor_tilts}")

        if not corridor_list:
            return

        if self.clamping:
            start_clamped = self.clamp_pose_to_corridor(self.initial_point, corridor_sequence[0])
            goal_clamped = self.clamp_pose_to_corridor(self.target_point, corridor_sequence[-1])
            start_clamped[2] = self.initial_angle
            goal_clamped[2] = self.target_angle or 0.0
        else:
            start_clamped = [self.initial_point[0], self.initial_point[1], self.initial_angle]
            goal_clamped = [self.target_point[0], self.target_point[1], self.target_angle or 0.0]

        try:
            t_start = time.time()
            # Waypoints intentionally left as None: the AP plans off the corridor list
            # and start/goal. Pass `waypoints[1:-1]` here if you want to test whether
            # via-points help the tangent construction.
            path, _, _, s = self.plan_motion_node.plan_motion(
                corridor_list,
                np.array(start_clamped),
                np.array(goal_clamped),
                None
            )
            self.get_logger().info(f"Path planning completed in {(time.time() - t_start) * 1000:.2f} ms")

            if path is not None and len(path) > 0:
                self.publish_planned_path(path)
            else:
                self.get_logger().warn("Planner solver failed (Empty Path).")
        except Exception as e:
            self.get_logger().warn(f"Trajectory generation error: {e}")

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def R(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def clamp_pose_to_corridor(self, point, corridor_id):
        poly = self.corridor_polygons[corridor_id]
        safe_poly = poly.buffer(-self.safety_margin)
        p = ShapelyPoint(point[0], point[1])
        if safe_poly.is_empty:
            node = self.G.nodes[corridor_id]
            return [node['pos'][0], node['pos'][1], 0.0]
        if safe_poly.contains(p):
            return [point[0], point[1], 0.0]
        nearest_pt = safe_poly.exterior.interpolate(safe_poly.exterior.project(p))
        return [nearest_pt.x, nearest_pt.y, 0.0]

    def _log_sequence(self, path, sequence, tilts):
        middle = [n for n in path if n not in ('start', 'end')]
        info = ", ".join(
            f"{n}={set(self.transition_graph.nodes[n]['corridors'])}"
            for n in middle if self.transition_graph.nodes[n].get('region')
        )
        self.get_logger().info(f"Dijkstra path ({len(middle)} middle nodes): {info}")
        self.get_logger().info(f"Corridor sequence ({len(sequence)}): {sequence}")

    def _save_graph_context(self):
        try:
            ctx = {
                "corridor_graph": {
                    "nodes": [
                        {"id": cid, "pos": [round(d['pos'][0], 4), round(d['pos'][1], 4)],
                         "width": round(d['width'], 4), "height": round(d['height'], 4),
                         "yaw": round(d['yaw'], 6)}
                        for cid, d in self.G.nodes(data=True)
                    ],
                    "edges": [list(e) for e in self.G.edges()],
                },
                "transition_graph": {
                    "nodes": [
                        ({"id": nid} if d.get('point') is None else
                         {"id": nid, "point": [round(d['point'][0], 4), round(d['point'][1], 4)],
                          "corridors": sorted(list(d['corridors'])),
                          "region": list(d['region']) if d.get('region') else None})
                        for nid, d in self.transition_graph.nodes(data=True)
                    ],
                    "edges": [[u, v, round(dd.get('weight', 0.0), 4)]
                              for u, v, dd in self.transition_graph.edges(data=True)],
                },
            }
            with open("graph_context.json", "w") as f:
                json.dump(ctx, f)
            self.get_logger().info("saved context to graph_context.json")
        except Exception as e:
            self.get_logger().warn(f"Could not save graph_context.json: {e}")

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    def publish_viz(self, waypoints, sequence):
        self.publish_waypoint_markers(waypoints)
        self.publish_path_marker(waypoints)
        self.publish_corridor_sequence_markers(sequence, self.G)
        self.publish_point_markers()

    def publish_planned_path(self, path):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for p in path:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = p[0], p[1]
            q = tf_transformations.quaternion_from_euler(0, 0, p[2])
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            msg.poses.append(pose)
        self.planned_path_publisher.publish(msg)

    def publish_waypoint_markers(self, waypoints):
        arr = MarkerArray()
        for i in range(50):
            arr.markers.append(Marker(action=Marker.DELETE, ns='path_waypoints', id=i))
        for i, wp in enumerate(waypoints):
            m = Marker(type=Marker.SPHERE, action=Marker.ADD, ns='path_waypoints', id=i)
            m.header.frame_id = self.map_frame
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wp[0], wp[1], 0.1
            m.color.r = float(1.0 - 0.5 * (i / max(len(waypoints), 1)))
            m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.8
            arr.markers.append(m)
        self.waypoint_marker_pub.publish(arr)

    def publish_path_marker(self, coords):
        m = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns='path', id=0)
        m.header.frame_id = self.map_frame
        m.scale.x = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.2, 0.1
        m.points = [Point(x=x, y=y) for x, y in coords]
        self.path_marker_pub.publish(m)

    def publish_point_markers(self):
        for i, (pt, c) in enumerate([(self.initial_point, (0.0, 0.0, 1.0)), (self.target_point, (1.0, 0.0, 0.0))]):
            m = Marker(type=Marker.SPHERE, action=Marker.ADD, ns='points', id=i)
            m.header.frame_id = self.map_frame
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pt[0], pt[1], 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = c[0], c[1], c[2], 1.0
            self.point_marker_pub.publish(m)

    def publish_corridor_sequence_markers(self, ids, G):
        arr = MarkerArray()
        for i in range(self.prev_corridor_marker_count):
            arr.markers.append(Marker(action=Marker.DELETE, ns='corridor_sequence', id=i))

        for i, cid in enumerate(ids):
            node = G.nodes[cid]
            m = Marker(type=Marker.CUBE, action=Marker.ADD, ns='corridor_sequence', id=i)
            m.header.frame_id = self.map_frame
            m.scale.x, m.scale.y, m.scale.z = node['width'], node['height'], 0.05
            t = i / max(len(ids) - 1, 1)
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2 + 0.6 * t, 0.0, 0.2 + 0.6 * (1 - t), 0.5
            m.pose.position.x, m.pose.position.y, m.pose.position.z = node['pos'][0], node['pos'][1], 0.025
            q = tf_transformations.quaternion_from_euler(0, 0, node['yaw'])
            m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            arr.markers.append(m)
        self.prev_corridor_marker_count = len(ids)
        self.corridor_marker_pub.publish(arr)

    def visualize_transition_points(self):
        arr = MarkerArray()
        for i, (nid, data) in enumerate(self.transition_graph.nodes(data=True)):
            if nid in ['start', 'end'] or 'point' not in data:
                continue
            m = Marker(type=Marker.SPHERE, action=Marker.ADD, ns='transition_points', id=i)
            m.header.frame_id = self.map_frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = data['point'][0], data['point'][1], 0.1
            m.color.a = 0.7
            if data['region']:
                m.scale.x, m.scale.y, m.scale.z = 0.25, 0.25, 0.25
                m.color.g, m.color.r, m.color.b = 1.0, 0.0, 0.0
            else:
                m.scale.x, m.scale.y, m.scale.z = 0.2, 0.2, 0.2
                m.color.b, m.color.r, m.color.g = 1.0, 0.0, 0.0
            arr.markers.append(m)
        self.transition_marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()