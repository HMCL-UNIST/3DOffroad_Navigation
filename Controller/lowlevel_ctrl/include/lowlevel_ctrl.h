

//  Software License Agreement (BSD License)
//  Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
//  All rights reserved.

//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:

//  1. Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.

//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ********************************************** 
//   @author: Hojin Lee <hojinlee@unist.ac.kr>
//   @date: September 10, 2022
//   @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
//   @brief: Low level acceleration control for ground vehicle in 3D terrain 
//   @details: control gazebo wheel vehicle given the acceraltion(ax), and steering angle (delta) / subscribe to joystick topic to control vehicle manually 



#include <sstream>
#include <string>
#include <list>
#include <queue>
#include <mutex> 
#include <thread> 
#include <numeric>
#include <boost/thread/thread.hpp>
#include <eigen3/Eigen/Geometry>

#include <ros/ros.h>
#include <ros/time.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Joy.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/Float64.h>
#include <autorally_msgs/chassisCommand.h>
#include <hmcl_msgs/vehicleCmd.h>

#include <dynamic_reconfigure/server.h>
#include <lowlevel_ctrl/testConfig.h>


#include "lowpass_filter.h"
#include "utils.h"



#define PI 3.14159265358979323846264338

class LowlevelCtrl 
{  
private:
ros::NodeHandle nh_ctrl_, nh_signal_;


std::string chassisCmd_topic, imu_topic;

std::mutex mtx_;
ros::Subscriber imuSub, accCmdSub, steerCmdSub, ctleffortSub, odomSub, joySub, deltaCmdSub;
ros::Publisher  filt_imu_pub, chassisCmdPub, debugPub, vel_x_pub, manual_ctrl_echo, target_vel_from_joy_pub, goal_pub;

dynamic_reconfigure::Server<lowlevel_ctrl::testConfig> srv;
dynamic_reconfigure::Server<lowlevel_ctrl::testConfig>::CallbackType f;


geometry_msgs::PoseStamped debug_msg;

nav_msgs::Odometry cur_odom;

double throttle_effort;
ros::Time throttle_effort_time;

Butterworth2dFilter imu_x_filter, imu_y_filter, imu_z_filter;
double filter_dt, cutoff_hz;
Butterworth2dFilter roll_filter, pitch_filter, yaw_filter; 
double rpy_dt,rpy_cutoff_hz;  
Butterworth2dFilter pitch_term_filter, rolling_term_filter;
double pitch_term, rolling_term;
double filt_x, filt_y, filt_z;

hmcl_msgs::vehicleCmd vehicle_cmd, prev_vehicle_cmd;
std::array<double,3> cur_rpy;

bool imu_received;
ros::Time state_time, prev_state_time;
double grav_accl = 9.806;
double roll_coef = 0.01035;
double cmd_scale, cmd_offset, brake_scale;
bool enforce_throttle;
double manual_acc_cmd, manual_throttle, manual_brake;
autorally_msgs::chassisCommand manual_cmd;
bool manual_ctrl;


public:
LowlevelCtrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj);
~LowlevelCtrl();

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
void ControlLoop();
void ImuCallback(const sensor_msgs::Imu::ConstPtr& msg);
void accCabllback(const hmcl_msgs::vehicleCmd::ConstPtr& msg);
void controleffortCallback(const std_msgs::Float64::ConstPtr& msg);
void joyCallback(const sensor_msgs::Joy::ConstPtr& msg);
void deltaCabllback(const std_msgs::Float64::ConstPtr& msg);


void dyn_callback(lowlevel_ctrl::testConfig& config, uint32_t level);



};



