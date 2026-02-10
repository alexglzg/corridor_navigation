import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # map_file = os.path.join(get_package_share_directory('corridor_bringup'), 
    #                         'maps/small_rooms', 'room_map_8.yaml')
    map_file = os.path.join(get_package_share_directory('corridor_bringup'), 
                            'maps/large_rooms', 'map_7.yaml')
    rviz_config_path = os.path.join(get_package_share_directory('corridor_bringup'), 
                                    'rviz', 'corridors.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('map_yaml', default_value=map_file),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{'yaml_filename': map_file}, {'frame_id': 'map'}]
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
            parameters=[{'robot_clearance': 0.3}]
        ),

        # Node(
        #     package='corridor_planner',
        #     executable='planner_node',
        #     name='corridor_planner',
        #     output='screen',
        #     parameters=[{
        #         'v_max': 1.0,         
        #         'robot_width': 0.430,      
        #         'robot_length': 0.508,    
        #         'robot_wheelbase': 0.4,
        #         'robot_delta_max': 0.5,
        #         'model_type': 'bicycle', 
        #         'sampling_dt': 0.100,    
        #         'map_frame': 'map'
        #     }]
        # ),

        Node(
            package='corridor_planner',
            executable='planner_node',
            name='corridor_planner',
            output='screen',
            parameters=[{
                'v_max': 0.5,             
                'omega_max': 2.0,         
                'robot_width': 0.34,      
                'robot_length': 0.237,    
                'model_type': 'unicycle', 
                'sampling_dt': 0.100,    
                'map_frame': 'map'
            }]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path]
        )
    ])