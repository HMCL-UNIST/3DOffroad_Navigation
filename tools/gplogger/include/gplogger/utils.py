
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
  @brief: util functions
  @details: util functions
"""

import math
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker

def b_to_g_rot(r,p,y):
    rot = np.array([[math.cos(p)*math.cos(y), -1*math.cos(p)*math.sin(y), math.sin(p)],
           [math.cos(r)*math.sin(y)+math.cos(y)*math.sin(r)*math.sin(p), math.cos(r)*math.cos(y)-math.sin(r)*math.sin(p)*math.sin(y), -math.cos(p)*math.sin(r)],
           [math.sin(r)*math.sin(y)-math.cos(r)*math.cos(y)*math.sin(p), math.cos(y)*math.sin(r)+math.cos(r)*math.sin(p)*math.sin(y), math.cos(r)*math.cos(p)]])
    return rot


def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """    
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

def predicted_distribution_traj_visualize(x_mean,x_var,y_mean,y_var,mean_predicted_state):
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
        marker_ref.pose.position.z = mean_predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state[i,2])             
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (255, 255, 0)
        marker_ref.color.a = 0.5                
        marker_ref.scale.x = 2*np.sqrt(x_var[i])
        marker_ref.scale.y = 2*np.sqrt(y_var[i])
        marker_ref.scale.z = 1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def predicted_trj_visualize(predicted_state):        
    marker_refs = MarkerArray() 
    for i in range(len(predicted_state[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predicted_state[i,0] 
        marker_ref.pose.position.y = predicted_state[i,1]              
        marker_ref.pose.position.z = predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (255, 255, 0)
        marker_ref.color.a = 0.5                
        marker_ref.scale.x = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.y = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.z = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_refs.markers.append(marker_ref)
        i+=1
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