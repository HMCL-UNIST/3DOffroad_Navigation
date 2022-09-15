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
  @brief: Torch version of util functions
"""


import math
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import torch
from hmcl_msgs.msg import Waypoints, Waypoint
import rospy 

def dist2d(point1, point2):
    dist2 = (point1[:,0] - point2[:,0])**2 + (point1[:,1] - point2[:,1])**2
    return torch.sqrt(dist2)

    
def b_to_g_rot(r,p,y):         
    row1 = torch.transpose(torch.stack([torch.cos(p)*torch.cos(y), -1*torch.cos(p)*torch.sin(y), torch.sin(p)]),0,1)
    row2 = torch.transpose(torch.stack([torch.cos(r)*torch.sin(y)+torch.cos(y)*torch.sin(r)*torch.sin(p), torch.cos(r)*torch.cos(y)-torch.sin(r)*torch.sin(p)*torch.sin(y), -torch.cos(p)*torch.sin(r)]),0,1)
    row3 = torch.transpose(torch.stack([torch.sin(r)*torch.sin(y)-torch.cos(r)*torch.cos(y)*torch.sin(p), torch.cos(y)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(r)*torch.cos(p)]),0,1)
    rot = torch.stack([row1,row2,row3],dim = 1)
    return rot


def wrap_to_pi(angle):
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def wrap_to_pi_torch(angle):

    return (((angle + torch.pi) % (2 * torch.pi)) - torch.pi)
    
    

def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
 
    q_norm = np.sqrt(np.sum(q ** 2))
 
    return 1 / q_norm * q

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    
    return unit_quat(np.array([qw, qx, qy, qz]))



def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = False):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel 


def traj_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        quat_tmp = euler_to_quaternion(0.0, 0.0, traj[i,3])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 255, 0)
        marker_ref.color.a = 0.2
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.2, 0.2, 0.15)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs

def predicted_distribution_traj_visualize(x_mean,x_var,y_mean,y_var,mean_predicted_state,color):
    marker_refs = MarkerArray() 
    for i in range(len(x_mean)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = x_mean[i]
        marker_ref.pose.position.y = y_mean[i]
        marker_ref.pose.position.z = mean_predicted_state[i,6]+0.2
        quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state[i,2])             
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5        
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        marker_ref.scale.x = 2*np.sqrt(x_var[i])
        marker_ref.scale.y = 2*np.sqrt(y_var[i])
        marker_ref.scale.z = 1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def path_to_waypoints(predicted_state):        
    wps = Waypoints() 
    wps.header.frame_id = "map"
    wps.header.stamp = rospy.Time.now()
    for i in range(len(predicted_state[:,0])):
        wp = Waypoint()                 
        wp.pose.pose.position.x = predicted_state[i,0] 
        wp.pose.pose.position.y = predicted_state[i,1]              
        wp.pose.pose.position.z = predicted_state[i,6]+0.2  
        quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        wp.pose.pose.orientation.w = quat_tmp[0]
        wp.pose.pose.orientation.x = quat_tmp[1]
        wp.pose.pose.orientation.y = quat_tmp[2]
        wp.pose.pose.orientation.z = quat_tmp[3]        
        wps.waypoints.append(wp)
        
    return wps

def predicted_trj_visualize(predicted_state,color):        
    marker_refs = MarkerArray() 
    marker_ref = Marker()
    marker_ref.header.frame_id = "map"  
    marker_ref.ns = "gplogger_ref"+str(0)
    marker_ref.id = 0
    marker_ref.type = Marker.LINE_STRIP
    marker_ref.action = Marker.ADD     
    marker_ref.scale.x = 0.05 
    for i in range(len(predicted_state[:,0])):                
        point_msg = Point()
        point_msg.x = predicted_state[i,0] 
        point_msg.y = predicted_state[i,1]              
        point_msg.z = predicted_state[i,6]+0.2         
        
        color_msg = ColorRGBA()
        color_msg.r = color[0]
        color_msg.g = color[1]
        color_msg.b = color[2]
        color_msg.a = 1.0
        marker_ref.points.append(point_msg)
        marker_ref.colors.append(color_msg)
    # marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
    # marker_ref.color.a = 0.5        
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        
        # marker_ref.scale.y = (i+1)/len(predicted_state[:,0])*0.1+0.1
        # marker_ref.scale.z = (i+1)/len(predicted_state[:,0])*0.1+0.1
    marker_refs.markers.append(marker_ref)
        
    return marker_refs


def ref_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states_"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 0, 255)
        marker_ref.color.a = 0.5
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.1, 0.1, 0.1)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs



def elips_predicted_path_distribution_traj_visualize(x_mean_set,x_var_set,y_mean_set,y_var_set,mean_predicted_state_set,color):
    marker_refs = MarkerArray() 
    for j in range(len(x_mean_set)):        
        for i in range(len(x_mean_set[j])):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "gplogger_ref_"+str(j)
            marker_ref.id = i*len(x_mean_set[j])
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD             
            marker_ref.scale.x = 2*np.sqrt(x_var_set[j][i])            
            marker_ref.scale.y = 2*np.sqrt(y_var_set[j][i])
            # marker_ref.scale.z = 0.1
            marker_ref.pose.position.x = x_mean_set[j][i]
            marker_ref.pose.position.y = y_mean_set[j][i]             
            marker_ref.pose.position.z = mean_predicted_state_set[i,j,6]+0.1       
            color_msg = ColorRGBA()
            if j == 0:
                color_msg.r = 233/255# 50
                color_msg.g = 16/255 # 229
                color_msg.b = 233/255 # 50
                color_msg.a = 0.7 # 0.5            
            else:
                color_msg.r = 191/255# 50
                color_msg.g = 245/255 # 229
                color_msg.b = 255/255 # 50
                color_msg.a = 0.7 # 0.5            
            marker_ref.color = color_msg
            
            
            # marker_ref.colors.append(color_msg)                           
            # marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (246, 229, 100 + 155/(len(x_mean_set)+0.01)*j)
            # marker_ref.color.a = 0.5        
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            
            
            marker_refs.markers.append(marker_ref)
            
    return marker_refs
def multi_predicted_distribution_traj_visualize(x_mean_set,x_var_set,y_mean_set,y_var_set,mean_predicted_state_set,color):
    marker_refs = MarkerArray() 
    for j in range(len(x_mean_set)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(j)
        marker_ref.id = j*len(x_mean_set[j])
        marker_ref.type = Marker.LINE_STRIP
        marker_ref.action = Marker.ADD 
        marker_ref.scale.x = 0.05 #2*np.sqrt(x_var_set[j][i])
        for i in range(len(x_mean_set[j])):
            point_msg = Point()
            point_msg = Point()
            point_msg.x = x_mean_set[j][i]
            point_msg.y = y_mean_set[j][i]             
            point_msg.z = mean_predicted_state_set[i,j,6]+0.2        
            color_msg = ColorRGBA()
            color_msg.r = color[0]# 50
            color_msg.g = color[1] # 229
            color_msg.b = color[2] # 50
            color_msg.a = color[3] # 0.5
            marker_ref.points.append(point_msg)
            marker_ref.colors.append(color_msg)                           
            # marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (246, 229, 100 + 155/(len(x_mean_set)+0.01)*j)
            # marker_ref.color.a = 0.5        
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            
            
        marker_refs.markers.append(marker_ref)
            
    return marker_refs


def predicted_trajs_visualize(mean_predicted_state_set,color):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[1]):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "mean_ref"+str(j)
        marker_ref.id = j
        marker_ref.type = Marker.LINE_STRIP
        marker_ref.action = Marker.ADD    
        marker_ref.scale.x = 0.05
        for i in range(mean_predicted_state_set.shape[0]):        
            
            point_msg = Point()
            point_msg.x = mean_predicted_state_set[i,j,0] 
            point_msg.y = mean_predicted_state_set[i,j,1]           
            point_msg.z = mean_predicted_state_set[i,j,6]+0.2         
            color_msg = ColorRGBA()
            color_msg.r = color[0]
            color_msg.g = color[1]
            color_msg.b = color[2]
            color_msg.a = color[3]
            marker_ref.points.append(point_msg)
            marker_ref.colors.append(color_msg)     
       
        marker_refs.markers.append(marker_ref)
         

    return marker_refs


def dist3d(point1, point2):
    

    x1, y1, z1 = point1[0:3]
    x2, y2, z2 = point2[0:3]

    dist3d = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    return math.sqrt(dist3d)

def gaussianKN2D(local_map_sizes, rsig=1.,csig=2.):
    """
    creates gaussian kernel with side length `rl,cl` and a sigma of `rsig,csig`
    """
    
    rl = local_map_sizes[0] # torch.max(local_map_sizes[:,0])
    cl = local_map_sizes[1] # torch.max(local_map_sizes[:,1])
    kernels = []    
   
    rx = torch.linspace(-(rl - 1) / 2., (rl - 1) / 2., rl).double().to(device= "cuda")        
    rsig = torch.transpose(rsig.repeat(len(rx),1),0,1)    
    rx = rx.repeat(len(rsig),1)
    
    cx = torch.linspace(-(cl - 1) / 2., (cl - 1) / 2., cl).double().to(device= "cuda")
    csig = torch.transpose(csig.repeat(len(cx),1),0,1)    
    cx = cx.repeat(len(csig),1)
    
    gauss_x = torch.exp(-0.5 * torch.square(rx) / torch.square(rsig))
    gauss_y = torch.exp(-0.5 * torch.square(cx) / torch.square(csig))    
    for i in range(len(csig)):
        kernel = torch.outer(gauss_x[i,:], gauss_y[i,:])
        kernel = kernel / (torch.sum(kernel)+1e-8)
        kernels.append(kernel)        
    
    
    return kernels





def find_closest_idx_from_path(cur_x,prev_path,expected_distance):    
    prev_finish_idx = 1    
    dist_to_current_pose = np.sqrt(np.sum(np.power((prev_path[:,:2]-cur_x[:2]),2),axis = 1))
    min_dist_to_current_idx = np.argmin(dist_to_current_pose)
    cum_dist = 0.0
    for i in range(min_dist_to_current_idx,len(dist_to_current_pose[min_dist_to_current_idx:])):
        cum_dist = cum_dist + dist_to_current_pose[i]
        if cum_dist >= expected_distance:
            prev_finish_idx = i
            break            
    if prev_finish_idx == min_dist_to_current_idx:
        prev_finish_idx+=1
    return min_dist_to_current_idx, prev_finish_idx   

def filt_path(path,max_length):    
    cum_dist = 0
    for i in range(1,len(path)):
        dist_tmp = np.sqrt(np.sum(np.power((path[i,:2]-path[i-1,:2]),2),axis = 0))
        cum_dist += dist_tmp
        if cum_dist > max_length:
            return path[:i,:]
        
    return path


