import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2

from corridor_gen.core.linemap import LineMap
from corridor_gen.core.graph import Graph

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA, Empty
import math
from corridor_navigation_msgs.msg import Corridor, Edge as EdgeMsg, Graph as GraphMsg
from corridor_navigation_msgs.srv import GetGraph

class CorridorGenerator(Node):
    def __init__(self):
        super().__init__('corridor_generator_node')

        self.declare_parameter('map_threshold', 250)
        self.declare_parameter('robot_clearance', 0.3)
        self.declare_parameter('debug_mode', False)

        self.threshold = self.get_parameter('map_threshold').get_parameter_value().integer_value
        self.robot_clearance = self.get_parameter('robot_clearance').get_parameter_value().double_value
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/floorplan_markers', 10)        
        self.floorplan_update_pub = self.create_publisher(Empty, '/floorplan_updated', 10)
        self.number_of_plotted_rectangles = 0

        self.graph_service = self.create_service(GetGraph, 'get_graph', self.handle_get_graph)
        self.rectangles = Graph() 
        self.img_height = None  
        self.origin = None  
        self.resolution = None
        self.id_to_rectangle_info = {}  
        self.rectangle_to_id = {}      


        # self.map_data = None
        self.get_logger().info("Corridor generator node started. Waiting for /map...")

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        self.img_height = height
        self.origin = origin
        self.resolution = resolution

        image = np.zeros((height, width), dtype=np.uint8)
        image[data == -1] = 127
        image[data == 0] = 255
        image[data == 100] = 0

        image = cv2.flip(image, 0)

        THRESHOLD = self.threshold  # PNG

        DEBUG = 1 if self.debug_mode else 0
        SHOW = 1 if self.debug_mode else 0
        SAVE = 0

        try:
            floor_plan = LineMap(image=image, threshold=THRESHOLD, debug=DEBUG, 
                                 resolution=resolution, robot_clearance=self.robot_clearance)
            floor_plan.process()

            self.clear_previous_markers()
            self.publish_floorplan_rectangles(floor_plan, height, resolution, origin)
            self.floorplan_update_pub.publish(Empty())
        
        except Exception as e:
            self.get_logger().error(f"Error in map_callback: {e}")
            self.clear_previous_markers() 
    
    def pixel_to_world(self, x_px, y_px, height_px, resolution, origin):
        """
        Convert pixel coordinates (OpenCV frame) to world coordinates (ROS frame)
        """
        x_m = x_px * resolution + origin[0]
        y_m = (height_px - y_px) * resolution + origin[1]
        return x_m, y_m

    def clear_previous_markers(self):
        if not hasattr(self, 'number_of_plotted_rectangles'):
            self.number_of_plotted_rectangles = 0

        marker_array = MarkerArray()

        for i in range(self.number_of_plotted_rectangles):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "floorplan"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

        self.number_of_plotted_rectangles = 0

    def publish_floorplan_rectangles(self, floor_plan, img_height, resolution, origin):
        if not hasattr(floor_plan, 'rectangles'):
            self.get_logger().warn("No rectangles found in floor plan.")
            return
        
        self.rectangles = floor_plan.rectangles
        self.id_to_rectangle_info = {}

        for node_id in self.rectangles.nodes():
            rect = node_id 
            x_px, y_px = rect.center
            x_m, y_m = self.pixel_to_world(x_px, y_px, img_height, resolution, origin)

            width = rect.half_extents[0] * 2 * resolution
            height = rect.half_extents[1] * 2 * resolution
            yaw = -rect.angle 

            self.id_to_rectangle_info[id(rect)] = {
                "center_x": x_m,
                "center_y": y_m,
                "width": width,
                "height": height,
                "yaw": yaw
            }

        marker_array = MarkerArray()

        self.get_logger().info(f"Publishing {len(floor_plan.rectangles.nodes())} floor plan rectangles...")
        for i, rect in enumerate(floor_plan.rectangles.nodes()):
            x_px, y_px = rect.center
            x_m, y_m = self.pixel_to_world(x_px, y_px, img_height, resolution, origin)

            width = rect.half_extents[0] * 2 * resolution
            height = rect.half_extents[1] * 2 * resolution

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "floorplan"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.scale.x = width
            marker.scale.y = height
            marker.scale.z = 0.05
            marker.pose.position.x = x_m
            marker.pose.position.y = y_m
            marker.pose.position.z = 0.025

            angle = -rect.angle
            marker.pose.orientation.z = math.sin(angle / 2)
            marker.pose.orientation.w = math.cos(angle / 2)

            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.4)

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        self.number_of_plotted_rectangles = len(marker_array.markers)

    def handle_get_graph(self, request, response):
        graph_msg = GraphMsg()
        self.rectangle_to_id = {}

        if not hasattr(self, 'rectangles'):
            self.get_logger().warn("No floor plan graph available.")
            return response

        for i, rect in enumerate(self.rectangles.nodes()):
            room = Corridor()

            x_px, y_px = rect.center
            x_m, y_m = self.pixel_to_world(
                x_px, y_px,
                self.img_height,
                self.resolution,
                self.origin
            )

            room.id = i
            room.center_x = x_m
            room.center_y = y_m
            room.width = rect.half_extents[0] * 2 * self.resolution
            room.height = rect.half_extents[1] * 2 * self.resolution
            room.yaw = -rect.angle
            graph_msg.nodes.append(room)
            self.rectangle_to_id[rect] = i 

        for u, v in self.rectangles.edges():
            edge = EdgeMsg()
            edge.from_corridor = self.rectangle_to_id[u]
            edge.to_corridor = self.rectangle_to_id[v]
            graph_msg.edges.append(edge)

        response.graph = graph_msg
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CorridorGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
