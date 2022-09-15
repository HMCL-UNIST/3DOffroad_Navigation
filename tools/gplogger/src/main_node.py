#!/usr/bin/env python

"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
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
  @brief: ROS node for recording training data of GP model.
  @details: inlcudeding extra tools for trajectory recording and visualization  
"""

from re import L
import rospy
import time
import threading
import numpy as np
import math 
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped 
from hmcl_msgs.msg import vehicleCmd
from autorally_msgs.msg import chassisState
from grid_map_msgs.msg import GridMap

from gplogger.vehicle_model import VehicleModel
from gplogger.utils import  get_odom_euler, get_local_vel,  wrap_to_pi
from gplogger.dataloader import DataLoader
# from gplogger.gp_model import GPModel
from gplogger.gpgridmap import GPGridMap
import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gplogger')

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class GPLoggerWrapper:
    def __init__(self):      
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)   
        self.gp_enable = rospy.get_param('~gp_enable', default=True)        
        self.gp_n_sample = rospy.get_param('~gp_n_sample', default=10)           
        self.gp_n_models = rospy.get_param('~gp_n_models', default=2)           
        self.gp_train_data_file = rospy.get_param('~gp_train_data_file', default="test_data.npz")             
        
        self.rollover_idx = rospy.get_param('~rollover_idx', default=0.6) # 0.6 from textbook           
        self.vehicle_width = rospy.get_param('~vehicle_width', default=0.25)           
        self.vehicle_height = rospy.get_param('~vehicle_height', default=0.25)           
        
        self.dt = self.t_horizon / self.n_nodes*1.0        
         # x, y, psi, vx, vy, wz, z ,  roll, pitch 
         # 0  1  2     3  4   5   6    7, 8   
        self.cur_x = np.transpose(np.zeros([1,9]))
        
        # Initialize Grid map. (including elevatio nand terrain type information)
        self.local_map = GPGridMap()  
        self.map_available = False
        
        # Initiazlie 3D vehicle model for state prediction 
        self.VehicleModel = VehicleModel(dt = self.dt, map_info = self.local_map)        
        # room for dataloaders  
        self.dataloaders = []
        # setup the number of GP models, and assign to each data loader
        for i in range(self.gp_n_models):            
            self.dataloaders.append(DataLoader(input_dim = 2, state_dim = len(self.cur_x), dt = 0.1, terrain_id = i ))
        
        self.odom_available   = False 
        self.vehicle_status_available = False         
        
        # Thread for optimization
        self.vehicleCmd = vehicleCmd()
        self._thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()        
        self.debug_msg = PoseStamped()
        
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd_manual"                    
        status_topic = "/is_data_busy"
        global_map_topic = "/traversability_estimation/global_map"        
        self.file_name = "gp_data"        
        self.logging = False
        # Publishers                
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.ctrl_sub = rospy.Subscriber(control_topic, vehicleCmd, self.ctrl_callback)
        self.data_saver_sub = rospy.Subscriber("/gp_data_save", Bool, self.data_saver_callback)
        self.data_logging_sub = rospy.Subscriber("/gp_data_logging", Bool, self.data_logging_callback)
        self.local_map_sub = rospy.Subscriber(global_map_topic, GridMap, self.gridmap_callback)        
        
        # logging callback 
        self.logging_timer = rospy.Timer(rospy.Duration(self.dt), self.logging_callback)         
        
        rate = rospy.Rate(4)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)            
            
            rate.sleep()


    """ 
    Predict state of vehicle one step ahead given the 3D vehicle model and the 3D terrain topology map with types of terrain.         
    """
    def run_prediction(self):                
        start = time.time()                 
        x = self.cur_x.copy()
        u = self.cur_u.copy()                
        
        if self.logging:            
            cur_states = np.concatenate([self.cur_x,u])                        
            predicted_states = np.concatenate([self.VehicleModel.dynamics_update(self.cur_x,u),u])
            
            pose_tmp= np.array([self.cur_x[0],self.cur_x[1]])
            terrain_type_tmp = int(self.local_map.get_terrain_type(pose_tmp))
            if terrain_type_tmp <= self.gp_n_models:
                self.dataloaders[terrain_type_tmp].append_state(cur_states,predicted_states)                     
                print_mgs = "recording process for "+ str(terrain_type_tmp) + " gp model ("+ str(self.dataloaders[terrain_type_tmp].n_data_set)+ ")"
                rospy.loginfo(print_mgs)
        

    """
    Record the error between the predictied states and the actual states of vehicle depending on the types of terrain.    
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
        # Do not save the data if the roll angle is too big. 
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
     
        def _thread_func():
            ########  Log State Data ###################################################
            self.run_prediction()   
            ########  Log State Data END ###################################################         
        self._thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self._thread.start()
        self._thread.join()
        

    """
    Listen to preprocessed map (including elevation and terrain type information)
    """
    def gridmap_callback(self,msg):    
        if self.map_available is False:
            self.map_available = True
        self.local_map.set_map(msg)
       
    """
    Listen to control input
    """
    def ctrl_callback(self,msg):
        self.vehicleCmd = msg

    """
    Trigger for logging data 
    """
    def data_logging_callback(self,msg):
        if self.map_available is False:
            print("map is not available to initiate logging")
            return
        if msg.data:
            self.logging = True
        else:
            self.logging = False

    """
    Trigger for saving data
    """
    def data_saver_callback(self,msg):
        if msg.data:
            messageToPring =  "Total number of GP models =  " + str(self.gp_n_models)
            rospy.loginfo(messageToPring)
            # gp_n_models Number of files are saved for gp_n_models of GP model training
            for i in range(self.gp_n_models):                
                save_path = pkg_dir + '/data/'+self.file_name+"_"+str(self.dataloaders[i].terrain_type)
                self.dataloaders[i].file_save(save_path)
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
    Listen to current odom 
    """
    def odom_callback(self, msg):       
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        

    
###################################################################################
def main():
    rospy.init_node("gplogger")    
    GPLoggerWrapper()
if __name__ == "__main__":
    main()