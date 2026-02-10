import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from corridor_navigation_interfaces.srv import GetGraph
import networkx as nx
import math
import tf_transformations
import numpy as np
from shapely.geometry import Polygon, Point as ShapelyPoint, LineString
from shapely.affinity import rotate, translate
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set
import time

from nav_msgs.msg import Path
from .core.plan_motion import PlanMotion
import arena
from arena import Corridor


@dataclass
class TransitionRegion:
    """Represents a transition region between two corridors."""
    corridor1_id: int
    corridor2_id: int
    polygon: Polygon  # The actual transition area
    representative_points: List[Tuple[float, float]]  # Key points for pathfinding


class PlannerNode(Node):
    def __init__(self):
        super().__init__('arena_planner')

        self.PLANNING_DT = 0.100

        ROSBOT_V_MAX = 0.5
        ROSBOT_OMEGA_MAX = 2.0

        # Graph structures
        self.graph = None
        self.G = None
        self.transition_graph = None
        self.corridor_polygons = {}
        self.transition_regions = {}  # (c1, c2) -> TransitionRegion
        
        # Path planning parameters - simplified to just distance
        self.corridor_change_penalty = 0.0  # Small penalty for changing corridors
        
        self.initial_point = (0.0, 0.0)
        self.target_point = None
        self.current_state = None
        self.initial_angle = 0.0
        self.target_angle = None

        # Service client
        self.cli = self.create_client(GetGraph, '/get_graph')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_graph service...')

        # Subscribers
        self.create_subscription(Empty, '/floorplan_updated', self.floorplan_updated_callback, 10)
        self.create_subscription(Point, '/initial_point', self.initial_point_callback, 10)
        self.create_subscription(Point, '/target_point', self.target_point_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        self.create_subscription(Float64MultiArray, '/rosbot2pro/state', self.state_listener_callback, 10)

        # Publishers
        self.path_marker_pub = self.create_publisher(Marker, 'p2p_path_marker', 1)
        self.point_marker_pub = self.create_publisher(Marker, 'p2p_point_markers', 2)
        self.corridor_marker_pub = self.create_publisher(MarkerArray, 'p2p_corridor_markers', 1)
        self.transition_marker_pub = self.create_publisher(MarkerArray, 'transition_markers', 1)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, 'p2p_waypoint_markers', 1)
        self.prev_corridor_marker_count = 0

        self.planned_path_publisher = self.create_publisher(Path, '/rosbot2pro/planned_path', 10)

        self.get_logger().info("PlannerNode initialized.")
        self.plan_motion_node = PlanMotion()
        self.plan_motion_node.initialize_planner(
            # vehicle = arena.Unicycle(state = [0]*3, width = 0.237, length = 0.237, v_max = ROSBOT_V_MAX, v_min = -ROSBOT_V_MAX, omega_max = ROSBOT_OMEGA_MAX, omega_min = -ROSBOT_OMEGA_MAX), sampling_time=self.PLANNING_DT
            vehicle = arena.Unicycle(state = [0]*3, width = 0.34, length = 0.237, v_max = ROSBOT_V_MAX, v_min = -ROSBOT_V_MAX, omega_max = ROSBOT_OMEGA_MAX, omega_min = -ROSBOT_OMEGA_MAX), sampling_time=self.PLANNING_DT
        )        

    def floorplan_updated_callback(self, _):
        self.get_logger().info("Received /floorplan_updated signal. Requesting graph...")
        req = GetGraph.Request()
        future = self.cli.call_async(req)
        future.add_done_callback(self.on_graph_response)

    def on_graph_response(self, future):
        try:
            response = future.result()
            graph_msg = response.graph
            self.graph = graph_msg
            
            # Start timing precomputation
            start_time = time.time_ns() // 1000000
            
            # Build the basic graph
            self.G = self.build_graph_from_msg(graph_msg)
            
            # Precompute transition regions and build transition graph
            self.compute_transition_regions()
            self.build_transition_graph()
            
            precomp_time = time.time_ns() // 1000000 - start_time 
             
            self.get_logger().info(f'Graph built with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.')
            self.get_logger().info(f'Created {len(self.transition_regions)} transition regions')
            # self.get_logger().info(f'Transition graph built with {len(self.transition_graph.nodes)} nodes in {precomp_time:.2f}s')
            # show time in milliseconds
            self.get_logger().info(f'Transition graph built with {len(self.transition_graph.nodes)} nodes in {precomp_time} ms')
            
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
        self.get_logger().info(f"Initial point set to: {self.initial_point}")

    def target_point_callback(self, msg):
        # publish as ros2 topic pub /target_point geometry_msgs/msg/Point "{x: 1.0, y: 2.0, z: 0.0}"
        self.target_point = (msg.x, msg.y)
        self.target_angle = 0.0
        self.get_logger().info(f"Target point set to: {self.target_point}")
        self.try_plan_path()

    def goal_pose_callback(self, msg):
        self.target_point = (msg.pose.position.x, msg.pose.position.y)
        self.target_angle = tf_transformations.euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])[2]
        self.get_logger().info(f"Target point set from RViz goal pose: {self.target_point}")
        self.try_plan_path()

    def initial_pose_callback(self, msg):
        self.initial_point = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.initial_angle = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])[2]
        self.get_logger().info(f"Initial point set from RViz estimated pose: {self.initial_point}")
        # if self.target_point is not None:
        #     self.try_plan_path()

    def create_corridor_polygon(self, cx, cy, width, height, yaw):
        """Create a Shapely polygon for a corridor."""
        half_w = width / 2
        half_h = height / 2
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        poly = Polygon(corners)
        poly = rotate(poly, yaw, origin=(0, 0), use_radians=True)
        poly = translate(poly, xoff=cx, yoff=cy)
        
        return poly

    def build_graph_from_msg(self, graph_msg):
        """Build basic corridor graph and cache polygons."""
        G = nx.Graph()
        
        # Add nodes and cache polygons
        valid_nodes = set()
        for corridor in graph_msg.nodes:
            node_id = corridor.id
            pos = (corridor.center_x, corridor.center_y)
            
            try:
                poly = self.create_corridor_polygon(
                    corridor.center_x, corridor.center_y,
                    corridor.width - 0.237, corridor.height - 0.237,  # Slightly shrink corridors for safety
                    corridor.yaw
                )
                
                self.corridor_polygons[node_id] = poly
                
                G.add_node(node_id, 
                          pos=pos,
                          width=corridor.width,
                          height=corridor.height,
                          yaw=corridor.yaw,
                          polygon=poly)
                valid_nodes.add(node_id)
                
            except Exception as e:
                self.get_logger().error(f"Failed to create corridor {node_id}: {e}")

        # Add edges - validate both endpoints exist
        valid_edges = 0
        invalid_edges = 0
        for edge in graph_msg.edges:
            src = edge.from_corridor
            dst = edge.to_corridor
            
            if src in valid_nodes and dst in valid_nodes:
                G.add_edge(src, dst)
                valid_edges += 1
            else:
                invalid_edges += 1
                missing = []
                if src not in valid_nodes:
                    missing.append(f"src={src}")
                if dst not in valid_nodes:
                    missing.append(f"dst={dst}")
                self.get_logger().warn(f"Skipping edge {src}-{dst}: missing {', '.join(missing)}")
        
        if invalid_edges > 0:
            self.get_logger().warn(f"Skipped {invalid_edges} invalid edges out of {len(graph_msg.edges)} total")

        return G

    def compute_transition_regions(self):
        """Compute transition regions between connected corridors."""
        self.get_logger().info(f"Computing transition regions for {len(self.G.edges())} edges")
        
        for edge in self.G.edges():
            c1, c2 = edge
            
            poly1 = self.corridor_polygons.get(c1)
            poly2 = self.corridor_polygons.get(c2)
            
            if poly1 is None or poly2 is None:
                self.get_logger().error(f"Missing polygon for corridor {c1 if poly1 is None else c2}")
                continue
            
            # Connected corridors should always overlap
            if not poly1.intersects(poly2):
                self.get_logger().error(f"Connected corridors {c1} and {c2} do not overlap - invalid graph edge")
                continue
                
            # Get intersection
            intersection = poly1.intersection(poly2)
            
            if intersection.is_empty:
                self.get_logger().error(f"Empty intersection between connected corridors {c1} and {c2}")
                continue
                
            if intersection.geom_type != 'Polygon':
                self.get_logger().warn(f"Non-polygon intersection between {c1} and {c2}: {intersection.geom_type}")
                # Try to handle other geometry types
                if hasattr(intersection, 'centroid'):
                    centroid = intersection.centroid
                    self.transition_regions[(c1, c2)] = TransitionRegion(
                        corridor1_id=c1,
                        corridor2_id=c2,
                        polygon=None,
                        representative_points=[(centroid.x, centroid.y)]
                    )
                continue
            
            # Valid polygon intersection
            overlap_area = intersection.area
            if overlap_area < 0.01:  # Very tiny intersection
                self.get_logger().warn(f"Extremely small overlap ({overlap_area:.4f}m²) between {c1} and {c2} - using centroid only")
                centroid = intersection.centroid
                self.transition_regions[(c1, c2)] = TransitionRegion(
                    corridor1_id=c1,
                    corridor2_id=c2,
                    polygon=intersection,
                    representative_points=[(centroid.x, centroid.y)]
                )
                continue
            
            # Generate representative points for the intersection
            points = self.generate_transition_points(intersection)
            
            self.transition_regions[(c1, c2)] = TransitionRegion(
                corridor1_id=c1,
                corridor2_id=c2,
                polygon=intersection,
                representative_points=points
            )
            
            self.get_logger().debug(f"Created transition region between {c1} and {c2}: "
                                  f"area={overlap_area:.1f}m², points={len(points)}")
        
        self.get_logger().info(f"Created {len(self.transition_regions)} transition regions")

    def generate_transition_points(self, intersection):
        """Generate strategic transition points within the intersection area."""
        points = []
        
        # Always include centroid as the primary transition point
        centroid = intersection.centroid
        points.append((centroid.x, centroid.y))
        
        bounds = intersection.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Add corner points if the intersection is large enough
        if width > 0.0 and height > 0.0:
            corners = [
                (bounds[0], bounds[1]),
                (bounds[2], bounds[1]),
                (bounds[2], bounds[3]),
                (bounds[0], bounds[3])
            ]
            for corner in corners:
                points.append(corner)
        
        # Add edge midpoints
        # edge_points = [
        #     ((bounds[0] + bounds[2])/2, bounds[1]),  # Bottom center
        #     ((bounds[0] + bounds[2])/2, bounds[3]),  # Top center
        #     (bounds[0], (bounds[1] + bounds[3])/2),  # Left center
        #     (bounds[2], (bounds[1] + bounds[3])/2)   # Right center
        # ]
        
        # for pt in edge_points:
        #     if intersection.contains(ShapelyPoint(pt)):
        #         points.append(pt)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for pt in points:
            pt_rounded = (round(pt[0], 3), round(pt[1], 3))
            if pt_rounded not in seen:
                seen.add(pt_rounded)
                unique_points.append(pt)
        
        return unique_points

    def build_transition_graph(self):
        """Build a graph connecting transition points with simple distance-based costs."""
        self.transition_graph = nx.DiGraph()
        
        # Add virtual start/end nodes
        self.transition_graph.add_node('start')
        self.transition_graph.add_node('end')
        
        # Track transition points by corridor
        points_by_corridor = {}
        
        # Add transition points as nodes
        node_counter = 0
        for (c1, c2), region in self.transition_regions.items():
            for pt in region.representative_points:
                node_id = f"t_{node_counter}"
                node_counter += 1
                
                self.transition_graph.add_node(
                    node_id,
                    point=pt,
                    corridors={c1, c2},  # Set of corridors this point belongs to
                    region=(c1, c2)
                )
                
                # Track points by corridor
                for c in [c1, c2]:
                    if c not in points_by_corridor:
                        points_by_corridor[c] = []
                    points_by_corridor[c].append(node_id)
        
        # Add corridor centers as additional nodes
        for corridor_id, data in self.G.nodes(data=True):
            node_id = f"c_{corridor_id}"
            self.transition_graph.add_node(
                node_id,
                point=data['pos'],
                corridors={corridor_id},
                region=None
            )
            
            if corridor_id not in points_by_corridor:
                points_by_corridor[corridor_id] = []
            points_by_corridor[corridor_id].append(node_id)
        
        # Connect nodes within corridors with simple Euclidean distance
        for corridor_id, node_list in points_by_corridor.items():
            for i, n1 in enumerate(node_list):
                for n2 in node_list[i+1:]:
                    pt1 = self.transition_graph.nodes[n1]['point']
                    pt2 = self.transition_graph.nodes[n2]['point']
                    
                    # Check if path is valid within corridor
                    if self.is_path_in_corridor(pt1, pt2, corridor_id):
                        # Simple Euclidean distance
                        dist = math.dist(pt1, pt2)
                        
                        # Add small penalty if this edge crosses corridor boundaries
                        # (i.e., both nodes are transition points between different corridors)
                        corridors1 = self.transition_graph.nodes[n1]['corridors']
                        corridors2 = self.transition_graph.nodes[n2]['corridors']
                        
                        if len(corridors1) > 1 and len(corridors2) > 1 and corridors1 != corridors2:
                            # This edge represents a corridor change
                            dist += self.corridor_change_penalty
                        
                        self.transition_graph.add_edge(n1, n2, weight=dist)
                        self.transition_graph.add_edge(n2, n1, weight=dist)
        
        self.visualize_transition_points()

    def is_path_in_corridor(self, pt1, pt2, corridor_id):
        """Check if straight line path between two points stays within corridor."""
        line = LineString([pt1, pt2])
        poly = self.corridor_polygons[corridor_id]
        
        # Check if line is fully contained within polygon
        if poly.contains(line):
            return True
        
        # If not fully contained, check how much is inside
        try:
            intersection = poly.intersection(line)
            if intersection.is_empty:
                return False
            
            # For safety, require the path to be FULLY within the corridor
            if hasattr(intersection, 'length') and hasattr(line, 'length'):
                # Only allow if 99.9% of the path is within the corridor (small tolerance for numerical errors)
                return intersection.length >= 0.999 * line.length
            else:
                return False
        except Exception as e:
            self.get_logger().warn(f"Error checking path in corridor: {e}")
            return False

    def find_corridors_containing_point(self, point):
        """Find all corridors containing a given point."""
        px, py = point
        shapely_point = ShapelyPoint(px, py)
        containing_corridors = []
        
        for corridor_id, poly in self.corridor_polygons.items():
            if poly.contains(shapely_point):
                containing_corridors.append(corridor_id)
        
        return containing_corridors
    
    def remove_redundant_corridors(self, corridor_sequence):
        """Remove redundant corridor switches like A->B->A patterns."""
        if len(corridor_sequence) <= 2:
            return corridor_sequence
        
        cleaned = []
        i = 0
        while i < len(corridor_sequence):
            # Add current corridor
            cleaned.append(corridor_sequence[i])
            
            # Look ahead for redundant patterns
            if i + 2 < len(corridor_sequence):
                if corridor_sequence[i] == corridor_sequence[i + 2]:
                    # Found A->B->A pattern
                    # Check if we can skip the middle corridor
                    # This is valid if A and the corridor after A (at i+3) are connected
                    if i + 3 < len(corridor_sequence):
                        if self.G.has_edge(corridor_sequence[i], corridor_sequence[i + 3]):
                            # Skip the redundant middle corridor
                            i += 2
                            continue
                    else:
                        # This is the end of the sequence, just skip the redundant part
                        i += 2
                        continue
            i += 1
        
        return cleaned

    def try_plan_path(self):
        # check initial time
        start_time_plan = time.time_ns() // 1000000

        """Plan path using simplified corridor selection."""
        if self.G is None or self.initial_point is None or self.target_point is None:
            self.get_logger().warn("Graph or points not set. Cannot plan path.")
            return

        # Find corridors containing start and goal
        start_corridors = self.find_corridors_containing_point(self.initial_point)
        goal_corridors = self.find_corridors_containing_point(self.target_point)

        if not start_corridors or not goal_corridors:
            self.get_logger().warn("Start or target point not inside any corridor.")
            return

        # self.get_logger().info(f"Start corridors: {start_corridors}")
        # self.get_logger().info(f"Goal corridors: {goal_corridors}")

        # Check if both points are in the same corridor
        common_corridors = set(start_corridors) & set(goal_corridors)
        if common_corridors:
            # Check direct path in any common corridor
            for corridor_id in common_corridors:
                if self.is_path_in_corridor(self.initial_point, self.target_point, corridor_id):
                    # Direct path is possible
                    self.get_logger().info(f"Direct path within corridor {corridor_id}")
                    waypoints = [self.initial_point, self.target_point]
                    self.publish_path_marker(waypoints)
                    self.publish_waypoint_markers(waypoints)
                    self.publish_point_markers()
                    self.publish_corridor_sequence_markers([corridor_id], self.G)

                    # Execute motion planning
                    self.execute_motion_planning([corridor_id], waypoints, None, True)
                    return

        # Need to use transition graph
        temp_graph = self.transition_graph.copy()
        
        # Add temporary start connections
        for corridor_id in start_corridors:
            candidates = []
            
            for node_id, data in temp_graph.nodes(data=True):
                if corridor_id in data.get('corridors', set()):
                    pt = data['point']
                    if self.is_path_in_corridor(self.initial_point, pt, corridor_id):
                        dist = math.dist(self.initial_point, pt)
                        candidates.append((node_id, dist))
            
            # Connect to nearest valid points
            candidates.sort(key=lambda x: x[1])
            # for node_id, dist in candidates[:3]:  # Connect to top 3 nearest
            for node_id, dist in candidates:  # Connect to top 3 nearest
                temp_graph.add_edge('start', node_id, weight=dist)
        
        # Add temporary end connections
        for corridor_id in goal_corridors:
            candidates = []
            
            for node_id, data in temp_graph.nodes(data=True):
                if corridor_id in data.get('corridors', set()):
                    pt = data['point']
                    if self.is_path_in_corridor(pt, self.target_point, corridor_id):
                        dist = math.dist(pt, self.target_point)
                        candidates.append((node_id, dist))
            
            candidates.sort(key=lambda x: x[1])
            # for node_id, dist in candidates[:3]:
            for node_id, dist in candidates:
                temp_graph.add_edge(node_id, 'end', weight=dist)

        try:
            # Find shortest path
            path = nx.shortest_path(temp_graph, 'start', 'end', weight='weight')
            path_cost = nx.shortest_path_length(temp_graph, 'start', 'end', weight='weight')
            
            # Extract waypoints and corridor sequence
            waypoints = [self.initial_point]
            # self.get_logger().info(f"Waypoints: {waypoints}")
            corridor_sequence = []
            # current_corridors = set(start_corridors)
            waypoint_corridor_mapping = []
            current_corridor = None
                        
            internal_nodes = path[1:-1]  # skip 'start' and 'end'
            S_list = []  # list of corridor sets for each internal node
            for node in internal_nodes:
                node_data = temp_graph.nodes[node]
                waypoints.append(node_data['point'])
                node_corridors = node_data['corridors']
                S_list.append(set(node_data.get('corridors', set())))
                waypoint_corridor_mapping.append(node_corridors)

            waypoints.append(self.target_point)

            start_set = set(start_corridors)
            goal_set = set(goal_corridors)

            # Choose initial corridor:
            if internal_nodes:
                first_S = S_list[0]
                inter = start_set & first_S
                if inter:
                    current_corridor = next(iter(inter))
                else:
                    # prefer a start corridor connected to any corridor in the first node
                    picked = None
                    for c in start_corridors:
                        for s in first_S:
                            if c == s or self.G.has_edge(c, s):
                                picked = c
                                break
                        if picked:
                            break
                    current_corridor = picked if picked else start_corridors[0]
            else:
                # No internal nodes (direct start->end use-case)
                inter = start_set & goal_set
                current_corridor = next(iter(inter)) if inter else start_corridors[0]

            corridor_sequence.append(current_corridor)

            # Iterate internal nodes with look-ahead to avoid skipping corridors
            for i, S_curr in enumerate(S_list):
                S_next = S_list[i + 1] if i + 1 < len(S_list) else goal_set

                if current_corridor in S_curr:
                    # We're inside current_corridor at this waypoint.
                    # If there is another corridor here that also appears next, switch to it now (prevents skipping).
                    candidates = (S_curr - {current_corridor}) & S_next
                    if candidates:
                        new_corridor = next(iter(candidates))
                        if corridor_sequence[-1] != new_corridor:
                            corridor_sequence.append(new_corridor)
                        current_corridor = new_corridor
                    # else: stay in current_corridor
                else:
                    # Must switch into a corridor that contains this waypoint.
                    # Prefer one that also appears in the next step.
                    candidates = S_curr & S_next
                    if candidates:
                        new_corridor = next(iter(candidates))
                    else:
                        # Next-best: one connected to the current corridor in the corridor graph
                        new_corridor = None
                        for c in S_curr:
                            if current_corridor is None or c == current_corridor or self.G.has_edge(current_corridor, c):
                                new_corridor = c
                                break
                        if new_corridor is None:
                            new_corridor = next(iter(S_curr))  # fallback

                    if corridor_sequence[-1] != new_corridor:
                        corridor_sequence.append(new_corridor)
                    current_corridor = new_corridor

            # Ensure we end in a goal corridor (without skipping)
            if goal_corridors and current_corridor not in goal_set:
                best_goal = None
                for gc in goal_corridors:
                    if gc == current_corridor or self.G.has_edge(current_corridor, gc):
                        best_goal = gc
                        break
                if best_goal is None:
                    best_goal = goal_corridors[0]
                if corridor_sequence[-1] != best_goal:
                    corridor_sequence.append(best_goal)

            # Clean up redundant corridor switches (A->B->A patterns)
            # corridor_sequence = self.remove_redundant_corridors(corridor_sequence)
            
            # self.get_logger().info(f"Path found with cost {path_cost:.2f}")
            self.get_logger().info(f"Corridor sequence: {corridor_sequence}")
            self.get_logger().info(f"Waypoints: {waypoints}")
            # log number of corridors
            self.get_logger().info(f"Corridor count: {len(corridor_sequence)}")
            # log path length
            path_length = 0.0
            for i in range(len(waypoints) - 1):
                path_length += math.dist(waypoints[i], waypoints[i + 1])
            self.get_logger().info(f"Path length: {path_length:.2f} m")
            # log path length in pixels (1 pixel = 0.01 m)
            self.get_logger().info(f"Path length: {path_length * 100:.0f} pixels")
            
            # Publish visualization
            self.publish_path_marker(waypoints)
            self.publish_waypoint_markers(waypoints)
            self.publish_point_markers()
            self.publish_corridor_sequence_markers(corridor_sequence, self.G)
            
            # Execute motion planning
            self.execute_motion_planning(corridor_sequence, waypoints, waypoint_corridor_mapping, False)
            
        except nx.NetworkXNoPath:
            self.get_logger().warn("No valid path found.")
            return
        
        total_time_plan = (time.time_ns() // 1000000) - start_time_plan
        self.get_logger().info(f"Total planning time: {total_time_plan:.3f} ms")

    def execute_motion_planning(self, corridor_sequence, waypoints, waypoint_corridor_mapping, one_corridor = False):
        """Execute motion planning using the simplified corridor sequence."""
        corridor_list = []
        if one_corridor:
            for corridor_id in corridor_sequence:
                node = self.G.nodes[corridor_id]
                width = node['height']
                height = node['width']
                center = [node['pos'][0], node['pos'][1]]
                tilt = node['yaw']
                
                corridor_n = Corridor.CorridorWorld(width=width, height=height, center=center, tilt=tilt)
                corridor_list.append(corridor_n)

        else:
            corridor_tilts = self.compute_corridor_tilts_from_waypoints(corridor_sequence, waypoints, waypoint_corridor_mapping)
            # corridor_tilts = self.check_first_corridor_tilt(corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilts)
            for i, corridor_id in enumerate(corridor_sequence):
                node = self.G.nodes[corridor_id]
                width = node['height']
                height = node['width']
                center = [node['pos'][0], node['pos'][1]]
                hw = np.array([height, width])
                height, width = np.abs(np.dot(self.R(corridor_tilts[i]), hw))
                tilt = node['yaw'] + corridor_tilts[i]
                # self.get_logger().info(f"Tilt for corridor {corridor_id}: {tilt:.2f} radians")
                corridor_n = Corridor.CorridorWorld(width=width, height=height, center=center, tilt=tilt)
                corridor_list.append(corridor_n)

        try: 
            x0, y0 = self.initial_point
            x1, y1 = self.target_point
            current_state = [x0, y0, self.initial_angle]
            desired_state = [x1, y1, self.target_angle if self.target_angle is not None else 0.0]
            
            # start_time_plan = time.time_ns() // 1000000
            self.path_planned, self.control_path_planned, self.trajectory_planned, _ = \
                self.plan_motion_node.plan_motion(corridor_list, np.array(current_state), np.array(desired_state), None)
                # self.plan_motion_node.plan_motion(corridor_list, np.array(current_state), np.array(desired_state), waypoints if not one_corridor else None)
            # total_time_plan = (time.time_ns() // 1000000) - start_time_plan
            # self.get_logger().info(f"Planning time: {total_time_plan:.3f} ms")
            
            # Send planned path to RViz
            self.publish_planned_path(self.path_planned)

        except:
            self.get_logger().warn("No valid trajectory found.")

    def R(self, theta):
        """Rotation matrix for angle theta."""
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    def check_first_corridor_tilt(self, corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilts):
        if (corridor_tilts[0] == corridor_tilts[1]):
            corridor_id = corridor_sequence[0]
            entrance_point = self.initial_point
            exit_point = self.find_corridor_exit_point(
                    corridor_id, 0, corridor_sequence, waypoints, waypoint_corridor_mapping, entrance_point
                )


            # Compute direction from entrance to exit
            if entrance_point and exit_point:
                dx = exit_point[0] - entrance_point[0]
                dy = exit_point[1] - entrance_point[1]
                
                # Snap to axis-aligned directions based on largest component
                if abs(dx) > abs(dy):
                    # Vertical movement dominates  
                    tilt = math.pi/2 if dy > 0 else -math.pi/2  # 90° (up) or -90°/270° (down)
                else:
                    # Horizontal movement dominates
                    tilt = 0.0 if dx > 0 else math.pi  # 0° (right) or 180° (left)
                
                # self.get_logger().warn(f"Corridor {corridor_id} corrected: entrance={entrance_point}, exit={exit_point}, tilt={math.degrees(tilt):.0f}°")
            else:
                # Fallback to original corridor orientation
                tilt = 0.0  # or self.G.nodes[corridor_id]['yaw']
                self.get_logger().warn(f"Could not determine entrance/exit for corridor {corridor_id}, using default tilt")
            
            corridor_tilts[0] = tilt
        
        return corridor_tilts
    
    def check_previous_tilt(self, corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilt):
        entrance_point = self.find_corridor_entrance_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping
                )
        exit_point = self.find_corridor_exit_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping, entrance_point
                )

        # Compute direction from entrance to exit
        if entrance_point and exit_point:
            dx = exit_point[0] - entrance_point[0]
            dy = exit_point[1] - entrance_point[1]
            
            # Snap to axis-aligned directions based on largest component
            if abs(dx) > abs(dy):
                # Vertical movement dominates  
                tilt = math.pi/2 if dy > 0 else -math.pi/2  # 90° (up) or -90°/270° (down)
            else:
                # Horizontal movement dominates
                tilt = 0.0 if dx > 0 else math.pi  # 0° (right) or 180° (left) 
            # self.get_logger().warn(f"Corridor {corridor_id} corrected: entrance={entrance_point}, exit={exit_point}, tilt={math.degrees(tilt):.0f}°")

            w_h_ratio = self.G.nodes[corridor_id]['width'] / self.G.nodes[corridor_id]['height']
            h_w_ratio = self.G.nodes[corridor_id]['height'] / self.G.nodes[corridor_id]['width']

            if w_h_ratio > 2.0:
                # Corridor is very wide - prefer horizontal orientation
                tilt = 0.0 if dx > 0 else math.pi
            elif h_w_ratio > 2.0:
                # Corridor is very tall - prefer vertical orientation
                tilt = math.pi/2 if dy > 0 else -math.pi/2
        else:
            # Fallback to original corridor orientation
            tilt = corridor_tilt  # or self.G.nodes[corridor_id]['yaw']
            self.get_logger().warn(f"Could not determine entrance/exit for corridor {corridor_id}, using previous tilt")
        return tilt
          
    
    def compute_corridor_tilts_from_waypoints(self, corridor_sequence, waypoints, waypoint_corridor_mapping):
        """
        Compute one tilt per corridor based on entrance and exit points.
        Each corridor gets an axis-aligned direction (0°, 90°, 180°, or 270°).
        """
        corridor_tilts = []
        
        # Determine entrance and exit points for each corridor
        entrance_point = None
        exit_point = None
        for i, corridor_id in enumerate(corridor_sequence):
            
            
            if i == 0:
                # First corridor: entrance is initial_point
                entrance_point = self.initial_point
                
                # Find exit point: last waypoint in this corridor or first waypoint in next corridor
                exit_point = self.find_corridor_exit_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping, entrance_point
                )
                
            elif i == len(corridor_sequence) - 1:
                # Find entrance point: first waypoint in this corridor or last waypoint from previous corridor
                entrance_point = self.find_corridor_entrance_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping
                )

                # Last corridor: exit is target_point
                exit_point = self.target_point
                
                
            else:
                # Middle corridors: find both entrance and exit
                entrance_point = self.find_corridor_entrance_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping
                )
                exit_point = self.find_corridor_exit_point(
                    corridor_id, i, corridor_sequence, waypoints, waypoint_corridor_mapping, entrance_point
                )
            
            # Compute direction from entrance to exit
            if entrance_point and exit_point:
                dx = exit_point[0] - entrance_point[0]
                dy = exit_point[1] - entrance_point[1]
                
                w_h_ratio = self.G.nodes[corridor_id]['width'] / self.G.nodes[corridor_id]['height']
                h_w_ratio = self.G.nodes[corridor_id]['height'] / self.G.nodes[corridor_id]['width']

                # Snap to axis-aligned directions based on largest component
                if abs(dx) > abs(dy):
                    # Horizontal movement dominates
                    tilt = 0.0 if dx > 0 else math.pi  # 0° (right) or 180° (left)
                else:
                    # Vertical movement dominates  
                    tilt = math.pi/2 if dy > 0 else -math.pi/2  # 90° (up) or -90°/270° (down)

                if i > 1 and (tilt == corridor_tilts[-1]) and (tilt == corridor_tilts[-2]):
                    # Same as previous corridors - try to pick the other axis if possible
                    if abs(dx) > abs(dy):
                        # Vertical movement dominates  
                        tilt = math.pi/2 if dy > 0 else -math.pi/2  # 90° (up) or -90°/270° (down)
                    else:
                        # Horizontal movement dominates
                        tilt = 0.0 if dx > 0 else math.pi  # 0° (right) or 180° (left)

                if (i == len(corridor_sequence) - 1) and (tilt == corridor_tilts[-1]):
                    # Snap to axis-aligned directions based on largest component
                    if abs(dx) > abs(dy):
                        # Vertical movement dominates  
                        tilt = math.pi/2 if dy > 0 else -math.pi/2  # 90° (up) or -90°/270° (down)
                    else:
                        # Horizontal movement dominates
                        tilt = 0.0 if dx > 0 else math.pi  # 0° (right) or 180° (left)

                if w_h_ratio > 2.0:
                    # Corridor is very wide - prefer horizontal orientation
                    tilt = 0.0 if dx > 0 else math.pi
                    if (i > 1) and (tilt == corridor_tilts[-1]) and (tilt == corridor_tilts[-2]):
                        corridor_tilts[-1] = self.check_previous_tilt(corridor_sequence[i-1], (i-1), corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilts[-1])
                        
                elif h_w_ratio > 2.0:
                    # Corridor is very tall - prefer vertical orientation
                    tilt = math.pi/2 if dy > 0 else -math.pi/2 
                    if (i > 1) and (tilt == corridor_tilts[-1]) and (tilt == corridor_tilts[-2]):
                        corridor_tilts[-1] = self.check_previous_tilt(corridor_sequence[i-1], (i-1), corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilts[-1])

                # self.get_logger().info(f"Corridor {corridor_id}: entrance={entrance_point}, exit={exit_point}, tilt={math.degrees(tilt):.0f}°")
            else:
                # Fallback to original corridor orientation
                tilt = 0.0  # or self.G.nodes[corridor_id]['yaw']
                self.get_logger().warn(f"Could not determine entrance/exit for corridor {corridor_id}, using default tilt")
            
            corridor_tilts.append(tilt)

            if i == 1:
                corridor_tilts = self.check_first_corridor_tilt(corridor_sequence, waypoints, waypoint_corridor_mapping, corridor_tilts)
        
        return corridor_tilts


    def find_corridor_entrance_point(self, corridor_id, corridor_index, corridor_sequence, waypoints, waypoint_mapping):
        """
        Find the entrance point for a corridor - the first waypoint in this corridor
        or the last waypoint from the previous corridor.
        """
        # Look for waypoints in this corridor
        waypoints_in_current = []
        for j, wp in enumerate(waypoints[1:-1]):  # Skip initial and target
            if corridor_id in waypoint_mapping[j]:
                waypoints_in_current.append((j, wp))
        
        if waypoints_in_current:
            # Return the first waypoint in this corridor
            return waypoints_in_current[0][1]
        
        # No waypoints in current corridor - look for last waypoint in previous corridor
        if corridor_index > 0:
            prev_corridor_id = corridor_sequence[corridor_index - 1]
            waypoints_in_prev = []
            
            for j, wp in enumerate(waypoints[1:-1]):
                if prev_corridor_id in waypoint_mapping[j]:
                    waypoints_in_prev.append((j, wp))
            
            if waypoints_in_prev:
                # Return the last waypoint from previous corridor
                return waypoints_in_prev[-1][1]
        
        # Fallback: use corridor center or initial point if first corridor
        if corridor_index == 0:
            return self.initial_point
        else:
            # Use the center of the current corridor
            return self.G.nodes[corridor_id]['pos']


    def find_corridor_exit_point(self, corridor_id, corridor_index, corridor_sequence, waypoints, waypoint_mapping, entrance_point):
        """
        Find the exit point for a corridor - the last waypoint in this corridor
        or the first waypoint in the next corridor.
        """
        # Look for waypoints in this corridor
        waypoints_in_current = []
        for j, wp in enumerate(waypoints[1:-1]):  # Skip initial and target
            if corridor_id in waypoint_mapping[j]:
                waypoints_in_current.append((j, wp))
        
        if waypoints_in_current:
            if entrance_point != waypoints_in_current[-1][1]:
                # Return the last waypoint in this corridor
                return waypoints_in_current[-1][1]
        
        # No waypoints in current corridor - look for first waypoint in next corridor
        if corridor_index < len(corridor_sequence) - 1:
            next_corridor_id = corridor_sequence[corridor_index + 1]
            
            for j, wp in enumerate(waypoints[1:-1]):
                if next_corridor_id in waypoint_mapping[j]:
                    # Return the first waypoint in next corridor
                    return wp
        
        # Fallback: use target point if last corridor or next corridor center
        if corridor_index == len(corridor_sequence) - 1:
            return self.target_point
        elif corridor_index < len(corridor_sequence) - 1:
            next_corridor_id = corridor_sequence[corridor_index + 1]
            return self.G.nodes[next_corridor_id]['pos']
        else:
            return self.G.nodes[corridor_id]['pos']

    def publish_planned_path(self, path):
        planned_path = Path()
        planned_path.header.frame_id = 'map'

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0

            yaw = point[2]
            quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            planned_path.poses.append(pose)

        self.planned_path_publisher.publish(planned_path)

    def visualize_transition_points(self):
        """Publish markers for transition points."""
        marker_array = MarkerArray()
        
        for i, (node_id, data) in enumerate(self.transition_graph.nodes(data=True)):
            if node_id in ['start', 'end'] or 'point' not in data:
                continue
            
            marker = Marker()
            marker.header.frame_id =  'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'transition_points'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            pt = data['point']
            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]
            marker.pose.position.z = 0.1
            
            # Color based on type
            corridors = data.get('corridors', set())
            if len(corridors) > 1:
                # Transition point between corridors - green
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.25
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                # Corridor center - blue
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            
            marker.color.a = 0.7
            marker_array.markers.append(marker)
        
        self.transition_marker_pub.publish(marker_array)

    def publish_waypoint_markers(self, waypoints):
        """Publish markers for the waypoints along the path."""
        # Clear old waypoint markers first
        clear_array = MarkerArray()
        for i in range(50):  # Clear up to 50 old waypoints
            marker = Marker()
            marker.header.frame_id =  'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'path_waypoints'
            marker.id = i
            marker.action = Marker.DELETE
            clear_array.markers.append(marker)
        
        if clear_array.markers:
            self.waypoint_marker_pub.publish(clear_array)
        
        # Create new waypoint markers
        marker_array = MarkerArray()
        
        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id =  'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'path_waypoints'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = 0.1
            
            # Color gradient along path
            t = i / max(len(waypoints) - 1, 1)
            marker.color.r = 1.0 - 0.5 * t
            marker.color.g = 0.5
            marker.color.b = 0.5 * t
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        if marker_array.markers:
            self.waypoint_marker_pub.publish(marker_array)

    def publish_path_marker(self, coords):
        marker = Marker()
        marker.header.frame_id =  'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'p2p_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.3

        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.2
        marker.color.a = 0.1

        marker.points = [Point(x=x, y=y, z=0.0) for x, y in coords]
        self.path_marker_pub.publish(marker)

    def publish_point_markers(self):
        for id_offset, (point, color) in enumerate([
            (self.initial_point, (0.0, 0.0, 1.0)),  # Blue
            (self.target_point, (1.0, 0.0, 0.0))   # Red
        ]):
            marker = Marker()
            marker.header.frame_id =  'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'p2p_points'
            marker.id = id_offset
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.05
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            self.point_marker_pub.publish(marker)

    def publish_corridor_sequence_markers(self, path_ids, G):
        marker_array = MarkerArray()

        # Clear old markers
        for i in range(self.prev_corridor_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id =  'map'
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = 'corridor_sequence'
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Add new markers with gradient coloring
        for i, corridor_id in enumerate(path_ids):
            node = G.nodes[corridor_id]
            
            marker = Marker()
            marker.header.frame_id =  'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'corridor_sequence'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.scale.x = node['width']
            marker.scale.y = node['height']
            marker.scale.z = 0.02

            # Simple gradient coloring
            t = i / max(len(path_ids) - 1, 1)
            marker.color.r = 0.2 + 0.6 * t
            marker.color.g = 0.0
            marker.color.b = 0.2 + 0.6 * (1 - t)
            marker.color.a = 0.5

            # Pose
            marker.pose.position.x = node['pos'][0]
            marker.pose.position.y = node['pos'][1]
            marker.pose.position.z = 0.0
            quat = tf_transformations.quaternion_from_euler(0, 0, node['yaw'])
            marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

            marker_array.markers.append(marker)

        self.prev_corridor_marker_count = len(path_ids)
        self.corridor_marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()