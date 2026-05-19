import sys
import os
from ament_index_python.packages import get_package_prefix

def _link_kappa_planner():
    """
    Dynamically links the standalone kappa-motion-planner library 
    from the ROS 2 workspace's src directory.
    """
    try:
        pkg_install_dir = get_package_prefix('corridor_planner')
        
        ws_root = os.path.abspath(os.path.join(pkg_install_dir, '..', '..'))
        
        kappa_path = os.path.join(ws_root, 'src', 'kappa-motion-planner', 'src')
        
        if os.path.exists(kappa_path) and kappa_path not in sys.path:
            sys.path.insert(0, kappa_path)
            
    except Exception as e:
        print(f"[corridor_planner] Warning: Could not automatically link kappa_planner: {e}")

# Execute the function immediately when the module loads
_link_kappa_planner()