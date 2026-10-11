cmake_minimum_required(VERSION 3.0.2)
project(ros_px4_aeroace)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  trajectory_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/aeroace_px4_offboard_node.py
  scripts/aeroace_reference_trajectory_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
