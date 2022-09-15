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
  @brief: Gridmap object
  @details: gridmap object 
"""

import numpy as np
import rospkg
import math 
import pandas as pd
from gplogger.utils import wrap_to_pi

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gplogger')

class GPGridMap:

    def __init__(self):
        self.map = None
        self.elev_map = None
        self.normal_vector = None
        self.right_corner_x = None
        self.right_corner_y = None
        self.map_resolution = None
        self.map_info = None
        self.c_size = None
        self.r_size = None
        self.surface_normal_x_idx = None
        self.surface_normal_y_idx = None
        self.surface_normal_z_idx = None
        self.elevation_idx = None

    def set_map(self,map):
        self.map = map 
        self.map_info = map.info   
        self.elevation_idx = self.map.layers.index("elevation")                
        self.c_size  = map.data[self.elevation_idx].layout.dim[0].size
        self.r_size  = map.data[self.elevation_idx].layout.dim[1].size 
        self.map_resolution = self.map_info.resolution 
        self.right_corner_x = self.map_info.pose.position.x + self.map_info.length_x/2
        self.right_corner_y = self.map_info.pose.position.y + self.map_info.length_y/2
        self.surface_normal_x_idx = self.map.layers.index("surface_normal_x")            
        self.surface_normal_y_idx = self.map.layers.index("surface_normal_y")            
        self.surface_normal_z_idx = self.map.layers.index("surface_normal_z")   
                   
        s = pd.Series(self.map.data[self.elevation_idx].data)
        sx = pd.Series(self.map.data[self.surface_normal_x_idx].data)
        sy = pd.Series(self.map.data[self.surface_normal_y_idx].data)
        sz = pd.Series(self.map.data[self.surface_normal_z_idx].data)

        self.map.data[self.elevation_idx].data = np.nan_to_num(self.map.data[self.elevation_idx].data, nan = 0.0)           
        self.normal_vector_x = np.nan_to_num(self.map.data[self.surface_normal_x_idx].data, nan = 0.0)   
        self.normal_vector_y = np.nan_to_num(self.map.data[self.surface_normal_y_idx].data, nan = 0.0)   
        self.normal_vector_z = np.nan_to_num(self.map.data[self.surface_normal_z_idx].data, nan = 1.0)   

        self.terrain_type_idx = self.map.layers.index("terrain_type")
        self.map.data[self.terrain_type_idx].data = np.nan_to_num(self.map.data[self.terrain_type_idx].data, nan = 0.0)                       
        

    def get_terrain_type_idx(self,idx):
        if self.terrain_type_idx is None:
            return 0
        if idx >= self.r_size*self.c_size:
            print("idx out of bound")            
            return 0       
        return self.map.data[self.terrain_type_idx].data[idx]
       
    def get_terrain_type(self,pose):
        if pose.shape[0] > 2:
            # return array of terrain type index
            return [self.get_terrain_type_idx(self.pose2idx(pose[i,:])) for i in range(len(pose))]            
        else:
            # single pose input 
            idx = self.pose2idx(pose)
            return self.get_terrain_type_idx(idx)

    def update_elevation_map(self):        
        self.elev_map = self.map.data[self.elevation_idx]       
        
    def update_normal_vectors(self):
        # up in z-axis                
        normal_x = np.nan_to_num(self.map.data[self.surface_normal_x_idx].data, nan = 0.0)
        normal_y = np.nan_to_num(self.map.data[self.surface_normal_y_idx].data,nan = 0.0)
        normal_z = np.nan_to_num(self.map.data[self.surface_normal_z_idx].data, nan=1.0)        
        self.normal_vector = np.array([normal_x,normal_y,normal_z])

    def normalize_vector(self,vector):
        return vector / np.sqrt(np.sum(vector**2))

    def get_normal_vector_idx(self,idx):
        assert self.surface_normal_x_idx is not None, "surface normal x is not available"
        
        if idx >= self.r_size*self.c_size:
            print("idx out of bound")            
            return np.array([0,0,1])
        else:            
            normal_vector = np.array([self.normal_vector_x[idx],self.normal_vector_y[idx],self.normal_vector_z[idx] ])
            assert np.sum(np.isnan(normal_vector)) < 1," nan"
            
            return self.normalize_vector(normal_vector)

    def get_normal_vector(self,pose):
        if self.map is None or self.surface_normal_x_idx is None:
            return 
        idx = self.pose2idx(pose)
        return self.get_normal_vector_idx(idx)                
    
    
    def get_elevation_idx(self,idx):
        if self.elevation_idx is None:
            return 
        if idx >= self.r_size*self.c_size:
            print("idx out of bound")            
            return
        else: 
            return self.map.data[self.elevation_idx].data[idx]
    
    def get_elevation(self,pose):
        idx = self.pose2idx(pose)
        if idx < 0:
            return 0.0
        return self.get_elevation_idx(idx)        

  
    def get_rollpitch(self,pose):
        if self.map is None:
            return 
        idx =self.pose2idx(pose)
        yaw = pose[2]
        if idx < 0:
            return 0.0, 0.0, yaw        
        if self.surface_normal_x_idx is None:
            return 0.0, 0.0, yaw        
        normal_vector = self.get_normal_vector_idx(idx)
        yaw_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0])        
        yaw_vec[2] = -1*(normal_vector[0]*yaw_vec[0]+normal_vector[1]*yaw_vec[1])/(normal_vector[2]+1e-10)
        yaw_vec = self.normalize_vector(yaw_vec)               
        ry = math.asin(yaw_vec[2])
        rz = math.acos(yaw_vec[0] / ( math.cos(ry)+1e-5))
        rx = math.acos(normal_vector[2]/ (math.cos(ry)+1e-5))
        roll = -1*rx
        pitch = -1*ry 
        yaw = rz 
        roll = wrap_to_pi(roll)
        pitch = wrap_to_pi(pitch)
        assert roll is not np.nan, "idx is out of bound"                    
        return roll, pitch, yaw
        

    def idx2pose(self,idx):        
        # top right is 0 - bottom left is last            
        assert idx < self.r_size*self.c_size, "idx is out of bound"                    
        grid_r = int(idx/(self.r_size))
        grid_c = (idx - grid_r*self.r_size)
        pose_x = self.map_info.pose.position.x+self.c_size/2*self.map_resolution-grid_c*self.map_resolution
        pose_y = self.map_info.pose.position.y+self.r_size/2*self.map_resolution-grid_r*self.map_resolution
        return [pose_x, pose_y]
        
    def pose2idx(self,pose):    
        grid_c_idx = (int)((self.right_corner_x - pose[0]) / self.map_resolution)
        grid_r_idx = (int)((self.right_corner_y - pose[1]) / self.map_resolution)
        if grid_c_idx >= self.c_size:
            return -1
        if grid_r_idx >= self.r_size:
            return -1        
        idx = grid_c_idx + grid_r_idx*self.r_size 
        if idx >= self.c_size*self.r_size:
            return -1                 
        return idx

    