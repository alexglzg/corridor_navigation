import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, ColorRGBA
from geometry_msgs.msg import Point, PoseStamped, Twist
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
import matplotlib.pylab as plt


import numpy as np
import arena
from math import pi, cos, sin

class PlanMotion():
    def __init__(self, logger):

        self.logger = logger
        self.corridor_list = None
        self.vehicle = None
        self.motion_planner = None

    def initialize_planner(self, vehicle, sampling_time = 0.050):
        self.vehicle = vehicle
        self.sampling_time = sampling_time
        
        phi1, phi2 = pi/2, pi/6
        height1, height2 = 6, 4
        width1, width2 = 0.431, 0.431
        width1, width2 = 3, 3

        start_point1 = [0, 0] 
        end_point1   = [start_point1[0] + height1 * cos(phi1),
                        start_point1[1] + height1 * sin(phi1)]
        start_point2 = [end_point1[0],
                        end_point1[1]]
        end_point2   = [start_point2[0] + height2 * cos(phi2),
                        start_point2[1] + height2 * sin(phi2)]

        corridor1 = arena.get_corridor_from_vector(start_point1, end_point1, width1, add_height = 0.2 * height1)
        corridor2 = arena.get_corridor_from_vector(start_point2, end_point2, width2, add_height = 0.2 * height2)

        self.motion_planner = arena.MotionPlanner(self.vehicle, [corridor1, corridor2], [0]*3, [0]*3)

        # self.get_logger().info("[PlanMotion] Planner initialized")

    def plan_motion(self, corridor_list, vehicle_start_pose, vehicle_end_pose, waypoint_list = None):
        if self.motion_planner is None:
            # self.get_logger().error("Planner not initialized, please run initialize_planner(...) first.")
            return

        debug = False
        self.motion_planner.update(vehicle = self.vehicle, corridor_list = corridor_list, start_pose = vehicle_start_pose, end_pose = vehicle_end_pose, waypoints = waypoint_list[1:-1] if waypoint_list is not None else None)

        if len(corridor_list) > 1:
            trajectory, intersection_detected = self.motion_planner.compute_trajectory_analytical(), False
            # self.get_logger().info(f"Analytical solution computed in {self.motion_planner.comp_time_analytical_sol*1000:.1f} ms")

            if debug:
                figure = arena.plot_corridors(corridor_list = corridor_list, linestyle = '--', color = 'gray', linewidth = 0.5)
                for trajectory_piece in trajectory:
                    trajectory_piece.plot_path(figure)
                    # Plot circumference
                    if isinstance(trajectory_piece, arena.CurvilinearArcUnicycle):
                        trajectory_piece.plot_circle(figure)

                for trajectory_piece in trajectory:
                    if isinstance(trajectory_piece, arena.TurnOnTheSpot):
                        print(f'\n\nTurn {trajectory_piece.turn_direction} for {trajectory_piece.maneuver_time}s of {trajectory_piece.delta_angle * 180 / pi} degrees ')
                
        elif len(corridor_list) == 1:
            trajectory, intersection_detected = self.motion_planner.compute_trajectory_ocp_one_corridor(), False
            t0 = trajectory.t0
            tf = trajectory.tf
            time_grid = np.arange(t0, tf, self.sampling_time)

            [x_ocp, y_ocp, theta_ocp, vs_ocp, omegas_ocp] = trajectory.sampler(trajectory.gist, time_grid)

            path = np.empty((0,3))
            control_path = np.empty((0,2))

            for i in range(len(x_ocp)):
                path = np.vstack((path, [x_ocp[i], y_ocp[i], theta_ocp[i]]))
                control_path = np.vstack((control_path, [vs_ocp[i], omegas_ocp[i]]))

            path = np.vstack((path, [vehicle_end_pose[0], vehicle_end_pose[1], vehicle_end_pose[2]])) 


            return path, control_path, trajectory, intersection_detected

        # else:
            # self.get_logger().error("You should provide at least one corridor to the planner")

        path = np.empty((0,3))
        control_path = np.empty((0,2))  

        if isinstance(trajectory, arena.UnicycleTrajectoryOptimal):
            t0 = trajectory.t0
            tf = trajectory.tf
            time_grid = np.arange(t0, tf, self.sampling_time)

            [x_ocp, y_ocp, theta_ocp, vs_ocp, omegas_ocp] = trajectory.sampler(trajectory.gist, time_grid)

            for i in range(len(x_ocp)):
                path = np.vstack((path, [x_ocp[i], y_ocp[i], theta_ocp[i]]))
                control_path = np.vstack((control_path, [vs_ocp[i], omegas_ocp[i]]))

            path = np.vstack((path, [vehicle_end_pose[0], vehicle_end_pose[1], vehicle_end_pose[2]])) 

            return path, control_path, trajectory, intersection_detected

        else:
            for trajectory_piece in trajectory:
                # Resample based on sampling time
                trajectory_piece.resample(new_samples_number = int(trajectory_piece.maneuver_time/self.sampling_time))

                pose_traj = np.hstack((trajectory_piece.path_coordinates[:,0:2], trajectory_piece.theta_trajectory.reshape((-1,1))))

                path = np.vstack((path, pose_traj))
                control_path = np.vstack((control_path, np.column_stack([trajectory_piece.forward_velocity, trajectory_piece.angular_velocity])))    

        return path, control_path, trajectory, intersection_detected