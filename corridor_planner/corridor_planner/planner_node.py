import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

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
    corridor1_id: int
    corridor2_id: int
    polygon: Polygon
    representative_points: List[Tuple[float, float]]

class PlannerNode(Node):
    def __init__(self):
        super().__init__('corridor_planner_node')

        # ---------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('v_max', 0.5),
                ('omega_max', 2.0),
                ('robot_width', 0.34),
                ('robot_length', 0.237),
                ('robot_wheelbase', 0.25),
                ('robot_delta_max', 0.5),
                ('model_type', 'unicycle'),
                ('sampling_dt', 0.100),
                ('map_frame', 'map')
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

        self.safety_margin = max(self.robot_width, self.robot_length) / 2.0 + 0.02
        self.clamping = False

        self.get_logger().info(
            f"Planner initialized: Model={self.model_type}"
        )

        # ---------------------------------------------------------
        # STATE VARIABLES
        # ---------------------------------------------------------
        self.graph = None
        self.G = None
        self.transition_graph = None
        self.corridor_polygons = {}
        self.transition_regions = {}
        self.corridor_change_penalty = 0.0
        
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
        
        # Instantiate PlanMotion
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
                    state=[0.0]*3, width=self.robot_width, length=self.robot_length,
                    v_max=self.v_max, v_min=-self.v_max,
                    omega_max=self.w_max, omega_min=-self.w_max
                )
            elif self.model_type == 'bicycle':
                return arena.Bicycle(
                    state=[0.0]*3, width=self.robot_width, length=self.robot_length,
                    wheelbase=self.robot_wheelbase, v_max=self.v_max, v_min=-self.v_max,
                    delta_max=self.delta_max, delta_min=-self.delta_max
                )
            else:
                self.get_logger().fatal(f"Unknown model_type: {self.model_type}")
                return None
        except Exception as e:
            self.get_logger().fatal(f"Failed to create vehicle: {e}")
            return None

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
            self.compute_transition_regions()
            self.build_transition_graph()
            self.get_logger().info(f'Graph built: {len(self.G.nodes)} nodes.')
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
        self.get_logger().info(f"Initial point set.")

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
        return G

    def compute_transition_regions(self):
        self.get_logger().info(f"Computing transition regions for {len(self.G.edges())} edges")
        for edge in self.G.edges():
            c1, c2 = edge
            poly1 = self.corridor_polygons.get(c1)
            poly2 = self.corridor_polygons.get(c2)
            if not poly1 or not poly2 or not poly1.intersects(poly2): continue
            
            intersection = poly1.intersection(poly2)
            if intersection.is_empty: continue
            
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

    def generate_transition_points(self, intersection):
        points = []
        centroid = intersection.centroid
        points.append((centroid.x, centroid.y))
        bounds = intersection.bounds
        if (bounds[2] - bounds[0]) > 0 and (bounds[3] - bounds[1]) > 0:
            points.extend([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])])
        
        seen, unique = set(), []
        for pt in points:
            pr = (round(pt[0], 3), round(pt[1], 3))
            if pr not in seen: seen.add(pr); unique.append(pt)
        return unique

    def build_transition_graph(self):
        self.transition_graph = nx.DiGraph()
        self.transition_graph.add_node('start'); self.transition_graph.add_node('end')
        
        points_by_corridor = {}
        nc = 0
        for (c1, c2), region in self.transition_regions.items():
            for pt in region.representative_points:
                nid = f"t_{nc}"; nc += 1
                self.transition_graph.add_node(nid, point=pt, corridors={c1, c2}, region=(c1, c2))
                for c in [c1, c2]: points_by_corridor.setdefault(c, []).append(nid)
        
        for cid, data in self.G.nodes(data=True):
            nid = f"c_{cid}"
            self.transition_graph.add_node(nid, point=data['pos'], corridors={cid}, region=None)
            points_by_corridor.setdefault(cid, []).append(nid)
        
        for cid, nodes in points_by_corridor.items():
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    pt1 = self.transition_graph.nodes[n1]['point']
                    pt2 = self.transition_graph.nodes[n2]['point']
                    if self.is_path_in_corridor(pt1, pt2, cid):
                        dist = math.dist(pt1, pt2)
                        if self.transition_graph.nodes[n1]['region'] and self.transition_graph.nodes[n2]['region']: dist += self.corridor_change_penalty
                        self.transition_graph.add_edge(n1, n2, weight=dist)
                        self.transition_graph.add_edge(n2, n1, weight=dist)
        self.visualize_transition_points()

    def is_path_in_corridor(self, pt1, pt2, corridor_id):
        line = LineString([pt1, pt2])
        poly = self.corridor_polygons[corridor_id]
        if poly.contains(line): return True
        try: return poly.intersection(line).length >= 0.999 * line.length
        except: return False

    def find_corridors_containing_point(self, point):
        return [cid for cid, poly in self.corridor_polygons.items() if poly.contains(ShapelyPoint(point[0], point[1]))]

    def try_plan_path(self):
        if self.G is None or not self.initial_point or not self.target_point: return

        start_cs = self.find_corridors_containing_point(self.initial_point)
        goal_cs = self.find_corridors_containing_point(self.target_point)

        if not start_cs or not goal_cs:
            self.get_logger().warn("Start or Goal outside corridors.")
            return

        common = set(start_cs) & set(goal_cs)
        if common:
            for cid in common:
                if self.is_path_in_corridor(self.initial_point, self.target_point, cid):
                    self.get_logger().info(f"Direct path in {cid}")
                    waypoints = [self.initial_point, self.target_point]
                    self.publish_viz(waypoints, [cid])
                    self.execute_motion_planning([cid], waypoints, None, True)
                    return

        temp = self.transition_graph.copy()
        
        for cid in start_cs:
            for nid, data in temp.nodes(data=True):
                if nid not in ['start','end'] and cid in data.get('corridors', set()):
                    if self.is_path_in_corridor(self.initial_point, data['point'], cid):
                        temp.add_edge('start', nid, weight=math.dist(self.initial_point, data['point']))
        
        for cid in goal_cs:
            for nid, data in temp.nodes(data=True):
                if nid not in ['start','end'] and cid in data.get('corridors', set()):
                    if self.is_path_in_corridor(data['point'], self.target_point, cid):
                        temp.add_edge(nid, 'end', weight=math.dist(data['point'], self.target_point))

        try:
            path = nx.shortest_path(temp, 'start', 'end', weight='weight')
            
            waypoints = [self.initial_point]
            corridor_sequence = []
            S_list = []
            
            for node in path[1:-1]:
                data = temp.nodes[node]
                waypoints.append(data['point'])
                S_list.append(set(data.get('corridors', set())))
            waypoints.append(self.target_point)

            current_c = next(iter(set(start_cs) & S_list[0])) if (path[1:-1] and (set(start_cs) & S_list[0])) else start_cs[0]
            corridor_sequence.append(current_c)

            for i, S_curr in enumerate(S_list):
                S_next = S_list[i+1] if i+1 < len(S_list) else set(goal_cs)
                if current_c in S_curr:
                    candidates = (S_curr - {current_c}) & S_next
                    if candidates:
                        current_c = next(iter(candidates))
                        if corridor_sequence[-1] != current_c: corridor_sequence.append(current_c)
                else:
                    current_c = next(iter(S_curr & S_next)) if (S_curr & S_next) else next(iter(S_curr))
                    if corridor_sequence[-1] != current_c: corridor_sequence.append(current_c)

            if goal_cs and current_c not in set(goal_cs):
                for gc in goal_cs:
                    if self.G.has_edge(current_c, gc):
                        corridor_sequence.append(gc); break
            
            self.publish_viz(waypoints, corridor_sequence)
            self.execute_motion_planning(corridor_sequence, waypoints, None, False)
            
        except nx.NetworkXNoPath:
            self.get_logger().warn("No path found.")

    def clamp_pose_to_corridor(self, point, corridor_id):
        poly = self.corridor_polygons[corridor_id]
        safe_poly = poly.buffer(-self.safety_margin)
        p = ShapelyPoint(point[0], point[1])
        if safe_poly.is_empty:
             node = self.G.nodes[corridor_id]
             return [node['pos'][0], node['pos'][1], 0.0]
        if safe_poly.contains(p): return [point[0], point[1], 0.0]
        nearest_pt = safe_poly.exterior.interpolate(safe_poly.exterior.project(p))
        return [nearest_pt.x, nearest_pt.y, 0.0]

    def execute_motion_planning(self, corridor_sequence, waypoints, waypoint_mapping, one_corridor=False):
        corridor_list = []
        if one_corridor:
            for cid in corridor_sequence:
                node = self.G.nodes[cid]
                corridor_list.append(Corridor.CorridorWorld(
                    width=node['height'], height=node['width'], center=[node['pos'][0], node['pos'][1]], tilt=node['yaw']
                ))
        else:
            corridor_tilts = self.compute_corridor_tilts_from_waypoints(corridor_sequence, waypoints, None)
            if len(corridor_sequence) > 1:
                corridor_tilts = self.check_first_corridor_tilt(corridor_sequence, waypoints, None, corridor_tilts)

            for i, cid in enumerate(corridor_sequence):
                node = self.G.nodes[cid]
                hw = np.array([node['width'], node['height']]) 
                h_final, w_final = np.abs(np.dot(self.R(corridor_tilts[i]), hw))
                corridor_list.append(Corridor.CorridorWorld(
                    width=w_final, height=h_final, 
                    center=[node['pos'][0], node['pos'][1]], 
                    tilt=node['yaw'] + corridor_tilts[i]
                ))

        if not corridor_list: return

        if self.clamping:
            start_clamped = self.clamp_pose_to_corridor(self.initial_point, corridor_sequence[0])
            goal_clamped = self.clamp_pose_to_corridor(self.target_point, corridor_sequence[-1])
            start_clamped[2] = self.initial_angle
            goal_clamped[2] = self.target_angle or 0.0
        else:
            start_clamped = [self.initial_point[0], self.initial_point[1], self.initial_angle]
            goal_clamped = [self.target_point[0], self.target_point[1], self.target_angle or 0.0]

        try:
            path, _, _, s = self.plan_motion_node.plan_motion(
                corridor_list, 
                np.array(start_clamped), 
                np.array(goal_clamped), 
                None
            )
            
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

    def compute_corridor_tilts_from_waypoints(self, sequence, waypoints, mapping):
        tilts = []
        for i, cid in enumerate(sequence):
            ent, ext = None, None
            if i == 0:
                ent = self.initial_point
                ext = self.find_corridor_exit_point(cid, i, sequence, waypoints)
            elif i == len(sequence)-1:
                ent = self.find_corridor_entrance_point(cid, i, sequence, waypoints)
                ext = self.target_point
            else:
                ent = self.find_corridor_entrance_point(cid, i, sequence, waypoints)
                ext = self.find_corridor_exit_point(cid, i, sequence, waypoints)
            
            if ent and ext:
                dx, dy = ext[0]-ent[0], ext[1]-ent[1]
                w_h = self.G.nodes[cid]['width'] / self.G.nodes[cid]['height']
                h_w = self.G.nodes[cid]['height'] / self.G.nodes[cid]['width']
                
                if abs(dx) > abs(dy): tilt = 0.0 if dx > 0 else math.pi
                else: tilt = math.pi/2 if dy > 0 else -math.pi/2
                
                if i > 1 and tilt == tilts[-1] == tilts[-2]:
                    if abs(dx) > abs(dy): tilt = math.pi/2 if dy > 0 else -math.pi/2
                    else: tilt = 0.0 if dx > 0 else math.pi
                
                if w_h > 2.0: tilt = 0.0 if dx > 0 else math.pi
                elif h_w > 2.0: tilt = math.pi/2 if dy > 0 else -math.pi/2
            else: tilt = 0.0
            tilts.append(tilt)
        return tilts

    def check_first_corridor_tilt(self, sequence, waypoints, mapping, tilts):
        if tilts[0] == tilts[1]:
            ent = self.initial_point
            ext = self.find_corridor_exit_point(sequence[0], 0, sequence, waypoints)
            if ent and ext:
                dx, dy = ext[0]-ent[0], ext[1]-ent[1]
                tilts[0] = math.pi/2 if dy > 0 else -math.pi/2 if abs(dx) <= abs(dy) else 0.0 if dx > 0 else math.pi
        return tilts

    def find_corridor_entrance_point(self, cid, idx, seq, wps):
        if idx > 0: return self.G.nodes[seq[idx-1]]['pos']
        return self.initial_point

    def find_corridor_exit_point(self, cid, idx, seq, wps):
        if idx < len(seq)-1: return self.G.nodes[seq[idx+1]]['pos']
        return self.target_point

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
            q = tf_transformations.quaternion_from_euler(0,0,p[2])
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            msg.poses.append(pose)
        self.planned_path_publisher.publish(msg)

    def publish_waypoint_markers(self, waypoints):
        arr = MarkerArray()
        for i in range(50): arr.markers.append(Marker(action=Marker.DELETE, ns='path_waypoints', id=i))
        for i, wp in enumerate(waypoints):
            m = Marker(type=Marker.SPHERE, action=Marker.ADD, ns='path_waypoints', id=i)
            m.header.frame_id = self.map_frame
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wp[0], wp[1], 0.1
            m.color.r = float(1.0 - 0.5*(i/max(len(waypoints),1)))
            m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.8
            arr.markers.append(m)
        self.waypoint_marker_pub.publish(arr)

    def publish_path_marker(self, coords):
        m = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns='path', id=0)
        m.header.frame_id = self.map_frame
        m.scale.x = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.2, 0.1
        m.points = [Point(x=x, y=y) for x,y in coords]
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
            t = i / max(len(ids)-1, 1)
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2 + 0.6*t, 0.0, 0.2 + 0.6*(1-t), 0.5
            m.pose.position.x, m.pose.position.y, m.pose.position.z = node['pos'][0], node['pos'][1], 0.025
            q = tf_transformations.quaternion_from_euler(0,0,node['yaw'])
            m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            arr.markers.append(m)
        self.prev_corridor_marker_count = len(ids)
        self.corridor_marker_pub.publish(arr)

    def visualize_transition_points(self):
        arr = MarkerArray()
        for i, (nid, data) in enumerate(self.transition_graph.nodes(data=True)):
            if nid in ['start', 'end'] or 'point' not in data: continue
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()