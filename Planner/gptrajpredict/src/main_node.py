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
  @brief: ROS node for learning based Uncertainty-aware path planning module.
  @details: uncertainty-aware path planning module  
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
from hmcl_msgs.msg import Waypoints, vehicleCmd
from visualization_msgs.msg import MarkerArray
from autorally_msgs.msg import chassisState
from grid_map_msgs.msg import GridMap


from gptrajpredict.vehicle_model_batch import VehicleModel
from gptrajpredict.utils_batch import find_closest_idx_from_path, predicted_trajs_visualize, multi_predicted_distribution_traj_visualize,  get_odom_euler, get_local_vel, predicted_trj_visualize, wrap_to_pi, path_to_waypoints, elips_predicted_path_distribution_traj_visualize
from gptrajpredict.gp_model import GPModel
from gptrajpredict.gpgridmap_batch import GPGridMap
from scipy.stats.distributions import norm

from dynamic_reconfigure.server import Server
from gptrajpredict.cfg import reconfigConfig

import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gptrajpredict')

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class GPTrajPredictWrapper:
    def __init__(self):        

        self.prediction_hz = rospy.get_param('~prediction_hz', default=3.0)
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)   
        self.gp_enable = rospy.get_param('~gp_enable', default=True)
        self.gp_sample_trj = rospy.get_param('~gp_sample_trj', default=True)                      
        self.gp_n_sample = rospy.get_param('~gp_n_sample', default= 20)                           
        self.input_random = rospy.get_param('~input_random', default=False)                      
        
        self.dt = self.t_horizon / self.n_nodes*1.0        
         # x, y, psi, vx, vy, wz, z ,  roll, pitch 
         # 0  1  2     3  4   5   6    7, 8   
        self.cur_x = np.transpose(np.zeros([1,9]))
        self.prev_path = None
        self.local_traj_msg = None
        self.prev_local_traj_msg = None
        self.local_traj_max_pathlength = 15
        

        # Initialize Grid map 
        self.local_map = GPGridMap(dt = self.dt)
        
        # Initialize GP Model 
        self.gpmodels = []
        if self.gp_enable:
            ## TO do, multiple gp generation and loading for specified number of terrain classes. 
            #  currently ony two types of terrain is supported, i.e., mud and grass area  
            self.GPmodel_grass = GPModel(dt = self.dt,  terrain_type = 0, data_file_name = "gp_data_0v4.npz", model_file_name = "GP_0v4.pth",model_load = True)                                           
            self.GPmodel_mud = GPModel(dt = self.dt,  terrain_type = 1, data_file_name = "gp_data_1v4.npz", model_file_name = "GP_1v4.pth",model_load = True)                                           
            self.gpmodels.append(self.GPmodel_grass)
            self.gpmodels.append(self.GPmodel_mud)
            self.VehicleModel = VehicleModel(dt = self.dt, N_node = self.n_nodes, map_info = self.local_map, gp_n_sample = self.gp_n_sample, gpmodels = self.gpmodels, gpmodel = self.GPmodel_grass, input_random = self.input_random)
        else:
            self.VehicleModel = VehicleModel(dt = self.dt,N_node = self.n_nodes,  map_info = self.local_map, gp_n_sample = self.gp_n_sample, input_random = self.input_random)
        
        # Initialize 3D vehicle model 
        self.local_map.vehicle_model = self.VehicleModel
        self.goal_pose = None
        self.odom_available   = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False 
        self.map_available = False
        
        # Thread for optimization
        self.vehicleCmd = vehicleCmd()
        self._thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        self.waypoint = PoseStamped()
        
        
        
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd_manual"                    
        status_topic = "/is_data_busy"
        global_map_topic = "/traversability_estimation/global_map"        
        goal_topic = "/move_base_simple/goal"         
        self.file_name = "test_data.npz"            
        sample_pred_traj_topic_name = "/sample_pred_trajectory"
        nominal_pred_traj_topic_name = "/nominal_pred_trajectory" 
        mean_pred_traj_topic_name = "/gpmean_pred_trajectory"         
        best_pred_traj_topic_name = "/best_gplogger_pred_trajectory" 
        local_traj_topic_name = "/local_traj"
        goal_topic = "/move_base_simple/goal"                
        # Publishers        
        self.local_traj_pub = rospy.Publisher(local_traj_topic_name, Waypoints, queue_size=2)        
        self.sample_predicted_trj_publisher = rospy.Publisher(sample_pred_traj_topic_name, MarkerArray, queue_size=2)        
        self.mean_predicted_trj_publisher    = rospy.Publisher(mean_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.nominal_predicted_trj_publisher = rospy.Publisher(nominal_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.best_predicted_trj_publisher = rospy.Publisher(best_pred_traj_topic_name, MarkerArray, queue_size=2)    
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        
        self.goal_pub = rospy.Publisher(goal_topic, PoseStamped, queue_size=2)    
        
        # Subscribers
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.ctrl_sub = rospy.Subscriber(control_topic, vehicleCmd, self.ctrl_callback)        
        self.local_map_sub = rospy.Subscriber(global_map_topic, GridMap, self.gridmap_callback)
               
        self.srv = Server(reconfigConfig, self.dyn_callback)
        self.path_planner_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.pathplanning_callback)         
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()
        
    def dyn_callback(self,config,level):        
        if config.set_goal:
            cur_goal = PoseStamped()            
            cur_goal.pose.position.x = config.goal_x
            cur_goal.pose.position.y = config.goal_y            
            self.goal_pub.publish(cur_goal)

        scale_data = {"dist_heuristic_cost_scale":  config.distance_heuristic_cost_scale,                         
                        "rollover_cost_scale": config.rollover_cost_scale, 
                        "kernel_dist_cost_scale": config.kernel_dist_cost_scale,                         
                        "prediction_diff_cost_scale" : config.prediction_diff_cost_scale}        
        self.local_map.set_scales(scale_data)
        return config

    def goal_callback(self,msg):        
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]

    def gridmap_callback(self,msg):                     
        if self.map_available is False:
            self.map_available = True
        self.local_map.set_map(msg)
                
    def ctrl_callback(self,msg):
        self.vehicleCmd = msg

    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        self.steering = -data.steering*25*math.pi/180
        
    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    def waypoint_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        self.waypoint = msg

 
    def pathplanning_callback(self,timer):        
        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return
        elif self.map_available is False:
            rospy.loginfo("Map is not available yet")
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
        self.run_prediction_sample()
        

    def run_prediction_sample(self):                  
        start = time.time()    
        if self.cur_x[3] > 100:
            if self.prev_path is not None:
                expected_distance = self.cur_x[3]*(1/self.prediction_hz)                                    
                # find closest waypoint from the previous path 
                prev_start_idx, prev_finish_idx = find_closest_idx_from_path(self.cur_x,self.prev_path,expected_distance)            
                # set x = the closest waypoint                                 
                x = self.cur_x.copy()                    
                x[:3] = self.prev_path[prev_finish_idx,:3].copy()
            else:
                x = self.cur_x.copy()                    
        else:
            x = self.cur_x.copy()     
        
        if self.input_random:
            self.VehicleModel.reset_input()                           

        gpsample_predictedStates ,gpmean_set_of_predstates, nominal_predictedStates = self.VehicleModel.batch_predict_multistep_multiinput_with_sample_gp(x,self.goal_pose)    
        
        x_pose_means_set = []
        x_pose_vars_set  = []
        y_pose_means_set = []        
        y_pose_vars_set  = []
        # extract trajectories with respecto to the combinations of inputs 
        for j in range(self.VehicleModel.uset.shape[0]):    
            x_pose_means = []
            x_pose_vars = []
            y_pose_means = []        
            y_pose_vars = []        
            for i in range(self.n_nodes):        
                x_pose_mean, x_pose_var  = norm.fit(gpsample_predictedStates[i,:,j,0])                
                y_pose_mean, y_pose_var  = norm.fit(gpsample_predictedStates[i,:,j,1])                 
                x_pose_means.append(x_pose_mean)
                x_pose_vars.append(x_pose_var)
                y_pose_means.append(y_pose_mean)
                y_pose_vars.append(y_pose_var)            
            x_pose_means_set.append(np.array(x_pose_means))
            x_pose_vars_set.append(np.array(x_pose_vars))
            y_pose_means_set.append(np.array(y_pose_means))
            y_pose_vars_set.append(np.array(y_pose_vars))            
        # predictedState = x, y, psi, vx, vy, wz, z, roll, pitch 
        #                  0  1  2     3  4   5   6 7,    8            
        x_pose_means_set = np.array(x_pose_means_set)
        x_pose_vars_set = np.array(x_pose_vars_set)
        y_pose_means_set = np.array(y_pose_means_set)
        y_pose_vars_set = np.array(y_pose_vars_set)

        best_path, costs = self.local_map.get_best_path(x_pose_means_set,y_pose_means_set,x_pose_vars_set,y_pose_vars_set,nominal_predictedStates[:,0,:,:],goal = self.goal_pose)                
        end=  time.time()           
        print("Total Time takes = : {:.5f}".format( end-start))        
        
        ### Publish local trajectory                         
        if self.cur_x[3] > 100:
            if self.prev_path is not None:
                # merge previous lane to computed lane            
                best_path = np.vstack([self.prev_path[prev_start_idx:prev_finish_idx+1,:],best_path]).copy()            
                  
        self.prev_path = best_path.copy()            
        self.local_traj_msg = path_to_waypoints(best_path) 
        self.local_traj_pub.publish(self.local_traj_msg)
        
        
        ######################### For plotting ####################################        
        best_path_color = [1,0,0]
        best_path_marker = predicted_trj_visualize(best_path,best_path_color)         
        self.best_predicted_trj_publisher.publish(best_path_marker)
        sample_path_color = [50,255,50,0.5]
        # pred_traj_marker = predicted_distribution_traj_visualize(x_pose_means_set,x_pose_vars_set,y_pose_means_set,y_pose_vars_set,nominal_predictedStates,sample_path_color)
        pred_traj_marker = multi_predicted_distribution_traj_visualize(x_pose_means_set,x_pose_vars_set,y_pose_means_set,y_pose_vars_set,nominal_predictedStates[:,0,:,:],sample_path_color)
        elip_pred_traj_marker = elips_predicted_path_distribution_traj_visualize(x_pose_means_set,x_pose_vars_set,y_pose_means_set,y_pose_vars_set,nominal_predictedStates[:,0,:,:],sample_path_color)        
        nominal_path_color = [0,1,0,1]
        nominal_pred_traj_marker = predicted_trajs_visualize(nominal_predictedStates[:,0,:,:],nominal_path_color)
        gpmean_path_color = [0,0,1,1]
        mean_pred_traj_marker = predicted_trajs_visualize(gpmean_set_of_predstates[:,0,:,:],gpmean_path_color)        
        self.mean_predicted_trj_publisher.publish(mean_pred_traj_marker)    
        self.nominal_predicted_trj_publisher.publish(nominal_pred_traj_marker) 
        # self.sample_predicted_trj_publisher.publish(pred_traj_marker)        
        self.sample_predicted_trj_publisher.publish(elip_pred_traj_marker)                       
        ######################### End plotting ####################################
        

###################################################################################
def main():
    rospy.init_node("gptrajpredict")    
    GPTrajPredictWrapper()
if __name__ == "__main__":
    main()




 
    


