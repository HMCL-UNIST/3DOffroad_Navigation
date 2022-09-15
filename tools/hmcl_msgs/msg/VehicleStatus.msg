Header header
hmcl_msgs/VehicleSteering steering_info
hmcl_msgs/VehicleSCC scc_info
hmcl_msgs/VehicleWheelSpeed wheelspeed
hmcl_msgs/VehicleGear gear_info
hmcl_msgs/VehicleLight light_info
hmcl_msgs/RemoteButton remote_button_info
# auto mode -> 0  off
# auto mode -> 1  on
uint8 auto_mode 
# ems mode -> 0  off
# ems mode -> 1  on
uint8 ems_mode 
float64 x_acceleration
float64 y_acceleration
float64 yaw_rate
