import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cmpe/functioning-bot/cooperative_perception_ros2/coop_perception_ws/install/robot_perception'
