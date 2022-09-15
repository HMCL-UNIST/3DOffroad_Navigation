
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


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
#include "lowlevel_ctrl.h"

using namespace std;
LowlevelCtrl::LowlevelCtrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_signal):  
  nh_ctrl_(nh_ctrl),
  nh_signal_(nh_signal),  
  filter_dt(0.005),
  cutoff_hz(10),
  rpy_dt(0.005),
  rpy_cutoff_hz(2),  
  cmd_scale(5.5),
  cmd_offset(-0.8),
  brake_scale(1),
  manual_acc_cmd(0.0),
  enforce_throttle(false),
  manual_throttle(0.0),
  manual_brake(0.0)
{
  imu_x_filter.initialize(filter_dt, cutoff_hz);
  imu_y_filter.initialize(filter_dt, cutoff_hz);
  imu_z_filter.initialize(filter_dt, cutoff_hz);

  roll_filter.initialize(rpy_dt, rpy_cutoff_hz);
  pitch_filter.initialize(rpy_dt, rpy_cutoff_hz);
  yaw_filter.initialize(rpy_dt, rpy_cutoff_hz);

  pitch_term_filter.initialize(rpy_dt, rpy_cutoff_hz);
  rolling_term_filter.initialize(rpy_dt, rpy_cutoff_hz);

  nh_signal_.param<std::string>("chassisCmd_topic", chassisCmd_topic, "/chassisCommand");
  nh_signal_.param<std::string>("imu_topic", imu_topic, "/imu/imu");  

  imuSub = nh_signal_.subscribe(imu_topic, 50, &LowlevelCtrl::ImuCallback, this);
  filt_imu_pub  = nh_signal_.advertise<sensor_msgs::Imu>("/filtered_imu", 2);      
  chassisCmdPub  = nh_ctrl.advertise<autorally_msgs::chassisCommand>(chassisCmd_topic, 2);   
  manual_ctrl_echo = nh_ctrl.advertise<hmcl_msgs::vehicleCmd>("/acc_cmd_manual",2);
  debugPub  = nh_ctrl.advertise<geometry_msgs::PoseStamped>("/lowlevel_debug", 2);   
  goal_pub  = nh_ctrl.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 2);   
    
  vel_x_pub =  nh_ctrl.advertise<std_msgs::Float64>("/state", 2);   
  ctleffortSub = nh_signal_.subscribe("/control_effort", 50, &LowlevelCtrl::controleffortCallback, this);   
  joySub = nh_signal_.subscribe("/joy_orig", 50, &LowlevelCtrl::joyCallback, this);   
  accCmdSub = nh_signal_.subscribe("/acc_cmd", 2, &LowlevelCtrl::accCabllback, this);   
  deltaCmdSub = nh_signal_.subscribe("/delta_cmd", 2, &LowlevelCtrl::deltaCabllback, this);      
  odomSub = nh_signal_.subscribe("/ground_truth/state", 50, &LowlevelCtrl::odomCallback, this);   

  boost::thread ControlLoopHandler(&LowlevelCtrl::ControlLoop,this);   
  ROS_INFO("Init Lowlevel Controller");
  
  f = boost::bind(&LowlevelCtrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);

}

LowlevelCtrl::~LowlevelCtrl()
{}

void LowlevelCtrl::deltaCabllback(const std_msgs::Float64::ConstPtr& msg){    
  vehicle_cmd.steering     = msg->data;
}

void LowlevelCtrl::joyCallback(const sensor_msgs::Joy::ConstPtr& msg){
    
    if(msg->buttons[5] > 0){
     vehicle_cmd.steering     = -1*msg->axes[3]*(25*PI/180.0);                              
    }

    if(msg->buttons[4] > 0 ){      
        manual_cmd.header = msg->header;
        vehicle_cmd.header = msg->header;
        vehicle_cmd.header.stamp = ros::Time::now();
        vehicle_cmd.acceleration = msg->axes[1]*3;
        
        manual_ctrl_echo.publish(vehicle_cmd); 
    }
}


void LowlevelCtrl::odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
  cur_odom = *msg;
  
     tf::Quaternion q_(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q_);
    m.getRPY(cur_rpy[0], cur_rpy[1], cur_rpy[2]);
    cur_rpy[0] = -cur_rpy[0];
    cur_rpy[1] = -cur_rpy[1];
    cur_rpy[2] = cur_rpy[2];

    cur_rpy[0] = roll_filter.filter(cur_rpy[0]);
    cur_rpy[1] = pitch_filter.filter(cur_rpy[1]);
    cur_rpy[2] = yaw_filter.filter(cur_rpy[2]);


}

void LowlevelCtrl::controleffortCallback(const std_msgs::Float64::ConstPtr& msg){
  
  vehicle_cmd.header.stamp = ros::Time::now();
  vehicle_cmd.acceleration = msg->data;
}

void LowlevelCtrl::ImuCallback(const sensor_msgs::Imu::ConstPtr& msg){  
  if(!imu_received){
    imu_received = true;
  }
  sensor_msgs::Imu filtered_imu_msg;
  filtered_imu_msg = *msg;
  
   // x - sin(pitch)*grav_accl 
  double accel_x_wihtout_gravity = msg->linear_acceleration.x -sin(cur_rpy[1])*grav_accl;
  // y + cos(pitch)sin(roll)*grav_accl
  double accel_y_wihtout_gravity = msg->linear_acceleration.y +cos(cur_rpy[1])*sin(cur_rpy[0])*grav_accl;
  // z - cos(pitch)cos(roll)*grav_accl
  double accel_z_wihtout_gravity = msg->linear_acceleration.z -cos(cur_rpy[1])*cos(cur_rpy[0])*grav_accl;
  filt_x = imu_x_filter.filter(accel_x_wihtout_gravity);
  filt_y = imu_y_filter.filter(accel_y_wihtout_gravity);
  filt_z = imu_z_filter.filter(accel_z_wihtout_gravity);
  filtered_imu_msg.linear_acceleration.x = filt_x;
  filtered_imu_msg.linear_acceleration.y = filt_y;
  filtered_imu_msg.linear_acceleration.z = filt_z;
  filt_imu_pub.publish(filtered_imu_msg);
  
}




void LowlevelCtrl::accCabllback(const hmcl_msgs::vehicleCmd::ConstPtr& msg){    
  vehicle_cmd.header = msg->header;
  vehicle_cmd.header.stamp = ros::Time::now();
  vehicle_cmd.acceleration = msg->acceleration;
  vehicle_cmd.steering     = msg->steering;
}


void LowlevelCtrl::ControlLoop()
{   double cmd_rate;
    ros::Rate loop_rate(50); // rate  
    while (ros::ok()){         
        auto start = std::chrono::steady_clock::now();                        
      if(imu_received){
        
        autorally_msgs::chassisCommand chassis_cmd;
        /////////////////// Control with Mapping //////////////////
        chassis_cmd.header = vehicle_cmd.header;                                
        double bias = 1.0;        
        double brake_bias = 1.102;
        double throttle_cmd = 0.0;
        double brake_cmd = 0.0;
        
        double compensated_pitch;         
        if (cur_rpy[1] >= 0 && cur_rpy[1] < 2*PI/180.0){
          compensated_pitch = 0.0;
        }        
        else{
          compensated_pitch = cur_rpy[1];
        }        
        pitch_term = grav_accl*sin(compensated_pitch);                        
        pitch_term = pitch_term_filter.filter(pitch_term);   
        
        rolling_term = roll_coef*grav_accl*cos(compensated_pitch);
        rolling_term = rolling_term_filter.filter(rolling_term);        
        
        
        double diff_time = fabs((vehicle_cmd.header.stamp - ros::Time::now()).toSec());        
        double local_speed = sqrt(pow(cur_odom.twist.twist.linear.x,2)+pow(cur_odom.twist.twist.linear.y,2)+pow(cur_odom.twist.twist.linear.z,2));
        std_msgs::Float64 speed_msg;
        speed_msg.data = local_speed;
        vel_x_pub.publish(speed_msg);              
        throttle_cmd = (vehicle_cmd.acceleration + rolling_term+pitch_term)/cmd_scale - cmd_offset/cmd_scale;                                 
        if(local_speed < 0.1 && vehicle_cmd.acceleration <=0){            
          throttle_cmd = 0.0;              
        }          
        if ( throttle_cmd < 0.0){
          brake_cmd = -throttle_cmd;
        }
        throttle_cmd = std::max(std::min(throttle_cmd,1.0),0.0);          
        brake_cmd = std::max(std::min(brake_cmd,1.0),0.0);          
        double steering_cmd = vehicle_cmd.steering/(25*PI/180.0);
        steering_cmd = std::max(std::min(steering_cmd,1.0),-1.0);          
        
          // throttle_cmd = manual_throttle;
        chassis_cmd.throttle = throttle_cmd;        
        chassis_cmd.frontBrake =brake_cmd;
        chassis_cmd.steering =steering_cmd;          
        chassisCmdPub.publish(chassis_cmd);
      }


     auto end = std::chrono::steady_clock::now();     
     loop_rate.sleep();
     std::chrono::duration<double> elapsed_seconds = end-start;
     if ( elapsed_seconds.count() > 1/cmd_rate){
       ROS_ERROR("computing control gain takes too much time");
       std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
     }
      
    }
}

void LowlevelCtrl::dyn_callback(lowlevel_ctrl::testConfig &config, uint32_t level)
{  
       
      if(config.set_goal){
        geometry_msgs::PoseStamped goal_pose;
        goal_pose.header.stamp = ros::Time::now();        
        goal_pose.pose.position.x = config.goal_x;      
        goal_pose.pose.position.y = config.goal_y; 
        goal_pose.pose.orientation.x = cur_odom.pose.pose.orientation.x;
        goal_pose.pose.orientation.y = cur_odom.pose.pose.orientation.y;
        goal_pose.pose.orientation.z = cur_odom.pose.pose.orientation.z;
        goal_pose.pose.orientation.w = cur_odom.pose.pose.orientation.w;        
        goal_pub.publish(goal_pose);
      }
        
    
}



int main (int argc, char** argv)
{
  ros::init(argc, argv, "LowlevelCtrl");
  
  ros::NodeHandle nh_ctrl, nh_signal;
  LowlevelCtrl LowlevelCtrl(nh_ctrl, nh_signal);

  ros::CallbackQueue callback_queue_ctrl, callback_queue_signal;
  nh_ctrl.setCallbackQueue(&callback_queue_ctrl);
  nh_signal.setCallbackQueue(&callback_queue_signal);
  

  std::thread spinner_thread_ctrl([&callback_queue_ctrl]() {
    ros::SingleThreadedSpinner spinner_ctrl;
    spinner_ctrl.spin(&callback_queue_ctrl);
  });

  std::thread spinner_thread_signal([&callback_queue_signal]() {
    ros::SingleThreadedSpinner spinner_signal;
    spinner_signal.spin(&callback_queue_signal);
  });

 

    ros::spin();

    spinner_thread_ctrl.join();
    spinner_thread_signal.join();


  return 0;

}
