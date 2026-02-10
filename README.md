# corridor_navigation

ROS 2 framework for fast, kinematically-feasible motion planning for non-holonomic autonomous mobile robots in structured environments. This repository implements the Automatic Corridor Generation method, which decomposes free space into a deterministic graph of overlapping rectangular corridors for efficient navigation.

This work corresponds to the paper:  
**"Fast Motion Planning for Non-Holonomic Mobile Robots via a Rectangular Corridor Representation of Structured Environments"**, accepted at **ICRA 2026**.

## Prerequisites

### System Requirements
* **ROS 2 Jazzy** (Ubuntu 24.04)
* **Nav2 Map Server**

### Python Dependencies
* `numpy`, `opencv-python`, `shapely`, `networkx`, `scipy`

### External Dependencies
This framework relies on the `arena` library for core geometric data structures and vehicle models.
> **Note**: The `arena` library is currently under construction for public release. To install the current development version, use:

```bash
pip install git+https://gitlab.kuleuven.be/u0153320/arena-framework.git@main

```

## Installation

1. **Create a workspace and clone:**

```bash
mkdir -p ~/corridor_ws/src
cd ~/corridor_ws/src
git clone [https://github.com/alexglzg/corridor_navigation.git](https://github.com/alexglzg/corridor_navigation.git)

```

2. **Install dependencies and build:**

```bash
cd ~/corridor_ws/src/corridor_navigation
pip install -r requirements.txt

cd ~/corridor_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

```

## Quick Start

To launch the complete pipeline (Map Server, Corridor Generator, Motion Planner, and RViz):

```bash
ros2 launch corridor_bringup full_system.launch.py

```

### Navigating in RViz

1. Wait for the **Map** and the **Corridor Graph** (colored rectangles) to appear.
2. Use the **"2D Pose Estimate"** toll in the RViz top bar to set an initial position.
3. Use the **"2D Goal Pose"** tool in the RViz top bar to set a destination.
4. The `corridor_planner` will automatically:
* Augment the graph with start/goal positions.
* Find the optimal corridor sequence.
* Generate and publish a kinematically-feasible trajectory.


## Repository Structure

* `corridor_navigation_interfaces`: Custom ROS 2 messages and services for the corridor graph.
* `corridor_gen`: The corridor generation engine and offline decomposition logic.
* `corridor_planner`: Online graph search and analytical motion planning.
* `corridor_bringup`: Centralized launch files, map datasets, and RViz configurations.

## Citation

An arXiv preprint of the ICRA 2026 paper will be linked here shortly.

```bibtex
@inproceedings{gonzalez2026fast,
  title={Fast Motion Planning for Non-Holonomic Mobile Robots via a Rectangular Corridor Representation of Structured Environments},
  author={Gonzalez-Garcia, Alejandro and Wyns, Sebastiaan and De Santis, Sonia and Swevers, Jan and Decr{\'e}, Wilm},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.