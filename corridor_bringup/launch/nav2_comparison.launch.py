import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bringup_dir = get_package_share_directory('corridor_bringup')
    
    map_yaml_file = os.path.join(bringup_dir, 'maps/large_rooms', 'map_7.yaml')
    params_file = os.path.join(bringup_dir, 'params', 'nav2_benchmarking_params.yaml')
    lattice_file = os.path.join(bringup_dir, 'params', 'lattices', 'output.json')

    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{'yaml_filename': map_yaml_file, 'use_sim_time': False}]
        ),
        
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[params_file, {
                'SmacLattice.lattice_filepath': lattice_file,
                'use_sim_time': False
            }]
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_comparison',
            parameters=[{'autostart': True, 'node_names': ['map_server', 'planner_server']}]
        )
    ])