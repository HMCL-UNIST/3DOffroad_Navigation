#!/usr/bin/env python

"""   
 Software License Agreement (BSD License)
  Copyright (c) 2022, Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>
  @date: September 10, 2022
  @copyright 2022 Ulsan National Institute of Science and Technology (UNIST)
  @brief: Tool for trajectory recording. 
  @details: to save the trajectory, save it by publishing "/gp_data_save" topic as true
"""
from re import L
import rospy
import numpy as np
import math 
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from hmcl_msgs.msg import vehicleCmd
from visualization_msgs.msg import MarkerArray, Marker
from autorally_msgs.msg import chassisState
from gplogger.utils import  get_odom_euler, get_local_vel,  wrap_to_pi
from gplogger.dataloader import DataLoader
import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gplogger')


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class TrajLoggerWrapper:
    def __init__(self):      
         # x, y, psi, vx, vy, wz, z, roll, pitch     
         # 0  1  2     3  4   5   6  7       8    
        self.cur_x = np.transpose(np.zeros([1,9]))        
        #################################################################        
        self.dataloader = DataLoader(input_dim = 2, state_dim = len(self.cur_x), dt = 0.2, terrain_id = 0)
        ### File name to save
        self.file_name = "traj_data"        
        ##        
        self.vehicleCmd = vehicleCmd()        
        self.chassisState = chassisState()
        self.odom = Odometry()                
        # Topics name 
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd_manual"                    
        status_topic = "/is_data_busy"
        # 
        self.odom_available   = False 
        self.vehicle_status_available = False                         
        # Publishers        
        self.path_history_marker = rospy.Publisher("/path_history", MarkerArray, queue_size=2)        
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)            
 
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.ctrl_sub = rospy.Subscriber(control_topic, vehicleCmd, self.ctrl_callback)
        self.data_saver_sub = rospy.Subscriber("/gp_data_save", Bool, self.data_saver_callback)
        
        # 10Hz logging callback 
        self.logging_timer = rospy.Timer(rospy.Duration(0.1), self.logging_callback)         
                                         
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)             
            rate.sleep()

    """ 
    Save trajectory states 
    """
    def logging_callback(self,timer):       
        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return

        current_euler = get_odom_euler(self.odom)
        for i in range(3):
            current_euler[i] = wrap_to_pi(current_euler[i])
        if(abs(current_euler[0]) > 80*math.pi/180):
            return
        
        local_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        cur_accel = self.vehicleCmd.acceleration        
        # x, y, psi, vx, vy, wz, z, roll, pitch 
        # 0  1  2     3  4   5   6 7,    8    \
        self.cur_x = np.transpose(np.array([self.odom.pose.pose.position.x,
                                            self.odom.pose.pose.position.y,
                                            current_euler[2],
                                            local_vel[0],
                                            local_vel[1],
                                            self.odom.twist.twist.angular.z,
                                            self.odom.pose.pose.position.z,                                                   
                                            current_euler[0],
                                            current_euler[1]]))
        self.cur_u = np.transpose(np.array([ self.steering,cur_accel]))
        
        ########  Log State Data ###################################################    
        if len(self.dataloader.Xstates) > 0:
            # log only if the distance from the previous recorded point greater than specific threshold (e.g. 0.2m or yaw difference greater than 10 degree)
            if math.sqrt((self.dataloader.Xstates[-1,0] - self.cur_x[0])**2 + (self.dataloader.Xstates[-1,1] - self.cur_x[1])**2) > 0.2 or abs(wrap_to_pi((self.dataloader.Xstates[-1,2] - current_euler[2])))> 10*math.pi/180.0:                     
                cur_states = np.concatenate([self.cur_x,self.cur_u])                            
                self.dataloader.append_state_for_traj(cur_states)                     
                print_mgs = "recording done : "+ " ("+ str(self.dataloader.n_data_set)+ " states)"
                rospy.loginfo(print_mgs)        
        else:
            #  Initial logging
            cur_states = np.concatenate([self.cur_x,self.cur_u])                        
            self.dataloader.append_state_for_traj(cur_states)                     
            print_mgs = "recordings done : "+ " ("+ str(self.dataloader.n_data_set)+ " states)"
            rospy.loginfo(print_mgs)   
        ########  Log State Data END ###################################################

        # Visualize recorded trajectory
        if len(self.dataloader.Xstates) > 0:
            pathmarkerArray = self.traj_to_linstrip(self.dataloader.Xstates)
            self.path_history_marker.publish(pathmarkerArray)   
###################################################################################


    """
    Generate Linstrip marker for RVIZ visualization 
    """
    def traj_to_linstrip(self,states_history):
        markers = MarkerArray()
        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = rospy.Time.now()
        line.type = 4
        line.scale.x = 0.05
        line.color.a = 1.0
        line.color.r = 1.0
        line.color.g = 0.0
        line.color.b = 0.0
        
        for i in range(len(states_history)):        
            p = Point()            
            p.x = states_history[i,0]
            p.y = states_history[i,1]
            p.z = states_history[i,6]+0.1            
            line.points.append(p)
        markers.markers.append(line)
        return markers

    """
    Listen to control action
    """
    def ctrl_callback(self,msg):
        self.vehicleCmd = msg

    """
    Subscriber for Data saving trigger 
    """
    def data_saver_callback(self,msg):
        if msg.data:
            save_path = pkg_dir + '/data/traj/'+self.file_name+"_"+str(self.dataloader.terrain_type)
            self.dataloader.file_save(save_path)
            messageToPring =  "file has been saved in " + str(save_path)
            rospy.loginfo(messageToPring)

    """
    Listen to current steering angle
    """
    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        self.steering = -data.steering*25*math.pi/180

    """
    Listen to current odom msg
    """   
    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    


def main():
    rospy.init_node("gplogger")    
    TrajLoggerWrapper()
if __name__ == "__main__":
    main()



