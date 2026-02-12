import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    bringup_dir = get_package_share_directory('corridor_bringup')
    map_file = os.path.join(bringup_dir, 
                            'maps/large_rooms', 'structured_map_20.yaml') 
    params_file = os.path.join(bringup_dir, 'params', 'corridor_system_params.yaml')
    
    rviz_config_path = os.path.join(bringup_dir, 
                                    'rviz', 'corridors.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('map_yaml', default_value=map_file, description='Path to the map yaml file'),
        DeclareLaunchArgument('params_file', default_value=params_file, description='Path to the ROS2 parameters file'),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[LaunchConfiguration('params_file'),
                        {'yaml_filename': LaunchConfiguration('map_yaml')}]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            parameters=[{'autostart': True}, {'node_names': ['map_server']}]
        ),

        Node(
            package='corridor_gen',
            executable='corridor_generator_node',
            name='corridor_generator',
            parameters=[LaunchConfiguration('params_file')]
        ),

        Node(
            package='corridor_planner',
            executable='planner_node',
            name='corridor_planner',
            output='screen',
            parameters=[LaunchConfiguration('params_file')]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path]
        )
    ])


# ros2 launch corridor_bringup full_system.launch.py map_yaml:=/home/alex/arena_ws/src/corridor_bringup/maps/small_rooms/structured_map_2.yaml
# ros2 launch corridor_bringup full_system.launch.py map_yaml:=/home/alex/arena_ws/src/corridor_bringup/maps/small_rooms/structured_rows.yaml
# ros2 launch corridor_bringup full_system.launch.py map_yaml:=/home/alex/arena_ws/src/corridor_bringup/maps/large_rooms/structured_rows_20.yaml
# ros2 launch corridor_bringup full_system.launch.py map_yaml:=/home/alex/arena_ws/src/corridor_bringup/maps/large_rooms/map_2.yaml