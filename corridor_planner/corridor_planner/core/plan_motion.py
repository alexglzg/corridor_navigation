import numpy as np
import arena
from math import pi, cos, sin
from typing import List, Any, Tuple

class PlanMotion:
    def __init__(self, logger):
        self.logger = logger
        self.vehicle = None
        self.motion_planner = None
        self.sampling_time = 0.1

    def initialize_planner(self, vehicle, sampling_time=0.1):
        """
        Initializes the Arena MotionPlanner with dummy corridors.
        """
        self.vehicle = vehicle
        self.sampling_time = sampling_time
        
        phi1, phi2 = pi/2, pi/6
        height1, height2 = 6, 4
        width1, width2 = 3, 3

        start_point1 = [0, 0] 
        end_point1   = [start_point1[0] + height1 * cos(phi1),
                        start_point1[1] + height1 * sin(phi1)]
        start_point2 = [end_point1[0],
                        end_point1[1]]
        end_point2   = [start_point2[0] + height2 * cos(phi2),
                        start_point2[1] + height2 * sin(phi2)]

        corridor1 = arena.get_corridor_from_vector(start_point1, end_point1, width1)
        corridor2 = arena.get_corridor_from_vector(start_point2, end_point2, width2)
        
        self.motion_planner = arena.MotionPlanner(self.vehicle, [corridor1, corridor2], [0]*3, [0]*3)
        self.logger.info("Arena MotionPlanner initialized.")

    def plan_motion(self, corridor_list, vehicle_start_pose, vehicle_end_pose, waypoint_list=None):
        """
        Updates the existing planner instance and computes trajectory.
        """
        if self.motion_planner is None:
            self.logger.error("Planner not initialized! Call initialize_planner first.")
            return None, None, None, False

        self.motion_planner.update(
            vehicle=self.vehicle, 
            corridor_list=corridor_list, 
            start_pose=vehicle_start_pose, 
            end_pose=vehicle_end_pose, 
            waypoints=waypoint_list[1:-1] if waypoint_list is not None else None
        )

        trajectory = None
        intersection_detected = False

        try:
            if len(corridor_list) > 1:
                trajectory = self.motion_planner.compute_trajectory_analytical()
                # TODO: Bicycle error: Trajectory generation error: 'list' object has no attribute 'maneuver_time'
                intersection_detected = False
            elif len(corridor_list) == 1:
                trajectory = self.motion_planner.compute_trajectory_ocp_one_corridor()
                #TODO: Check with Sonia why one corridor case is not available
                intersection_detected = False
            else:
                self.logger.error("No corridors provided.")
                return None, None, None, False

        except Exception as e:
            self.logger.error(f"Arena solver execution failed: {e}")
            return None, None, None, False

        # --- Resampling Logic ---
        path = np.empty((0, 3))
        control_path = np.empty((0, 2))

        if isinstance(trajectory, arena.UnicycleTrajectoryOptimal):
            t0 = trajectory.t0
            tf = trajectory.tf
            time_grid = np.arange(t0, tf, self.sampling_time)

            [x_ocp, y_ocp, theta_ocp, vs_ocp, omegas_ocp] = trajectory.sampler(trajectory.gist, time_grid)

            for i in range(len(x_ocp)):
                path = np.vstack((path, [x_ocp[i], y_ocp[i], theta_ocp[i]]))
                control_path = np.vstack((control_path, [vs_ocp[i], omegas_ocp[i]]))
                
        else:
            for trajectory_piece in trajectory:
                num_samples = int(trajectory_piece.maneuver_time / self.sampling_time)
                trajectory_piece.resample(new_samples_number=max(num_samples, 2))

                pose_traj = np.hstack((
                    trajectory_piece.path_coordinates[:, 0:2], 
                    trajectory_piece.theta_trajectory.reshape((-1, 1))
                ))

                path = np.vstack((path, pose_traj))
                
                if hasattr(trajectory_piece, 'forward_velocity') and hasattr(trajectory_piece, 'angular_velocity'):
                     control_chunk = np.column_stack([
                         trajectory_piece.forward_velocity, 
                         trajectory_piece.angular_velocity
                     ])
                     control_path = np.vstack((control_path, control_chunk))

        # Snap final point to goal
        path = np.vstack((path, [vehicle_end_pose[0], vehicle_end_pose[1], vehicle_end_pose[2]]))

        return path, control_path, trajectory, intersection_detected