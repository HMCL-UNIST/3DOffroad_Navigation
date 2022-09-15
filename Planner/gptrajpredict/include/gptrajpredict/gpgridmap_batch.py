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
  @brief: Compute uncertainty-aware traversability cost given the predictive path distributions
"""
from this import d
import numpy as np
import torch
from gptrajpredict.utils_batch import dist2d, gaussianKN2D, wrap_to_pi_torch

class GPGridMap:
    def __init__(self, device = "cuda", dt = 0.1):
        self.dt = dt
        self.vehicle_model = None        
        self.map = None
        self.elev_map = None
        self.normal_vector = None
        self.right_corner_x = None
        self.right_corner_y = None
        self.map_resolution = None
        self.map_info = None
        self.c_size = None
        self.r_size = None
        self.kernel_size = 5
        self.surface_normal_x_idx = None
        self.surface_normal_y_idx = None
        self.surface_normal_z_idx = None
        self.elevation_idx = None        
        
        self.rollover_cost_scale = 1.0
        self.dist_heuristic_cost_scale = 1.0
        
        self.kernel_dist_cost_scale = 1.0
        self.prediction_diff_cost_scale = 1.0
        
        self.torch_device  = device

    def set_scales(self,scale_data):
        self.rollover_cost_scale         = scale_data['rollover_cost_scale']        
        self.dist_heuristic_cost_scale   = scale_data['dist_heuristic_cost_scale']        
        
        self.kernel_dist_cost_scale      = scale_data['kernel_dist_cost_scale']        
        self.prediction_diff_cost_scale      = scale_data['prediction_diff_cost_scale']        

    def set_map(self,map):
        self.map = map 
        self.map_info = map.info           
        self.elevation_idx = self.map.layers.index("elevation")                
        self.geo_traversability_idx = self.map.layers.index("terrain_traversability")    

        self.c_size =torch.tensor(map.data[self.elevation_idx].layout.dim[0].size).to(device=self.torch_device)             
        self.r_size = torch.tensor(map.data[self.elevation_idx].layout.dim[1].size ).to(device=self.torch_device)  
        self.map_resolution = torch.tensor(self.map_info.resolution).to(device=self.torch_device)    
         
        self.right_corner_x = torch.tensor(self.map_info.pose.position.x + self.map_info.length_x/2).to(device=self.torch_device)          
        self.right_corner_y = torch.tensor(self.map_info.pose.position.y + self.map_info.length_y/2).to(device=self.torch_device)  
        
        
        self.surface_normal_x_idx = self.map.layers.index("surface_normal_x")            
        self.surface_normal_y_idx = self.map.layers.index("surface_normal_y")            
        self.surface_normal_z_idx = self.map.layers.index("surface_normal_z")   
            
        self.map.data[self.geo_traversability_idx].data = np.nan_to_num(self.map.data[self.geo_traversability_idx].data, nan = 0.0)           
        self.map.data[self.elevation_idx].data = np.nan_to_num(self.map.data[self.elevation_idx].data, nan = 0.0)           
        self.normal_vector_x = np.nan_to_num(self.map.data[self.surface_normal_x_idx].data, nan = 0.0)   
        self.normal_vector_y = np.nan_to_num(self.map.data[self.surface_normal_y_idx].data, nan = 0.0)   
        self.normal_vector_z = np.nan_to_num(self.map.data[self.surface_normal_z_idx].data, nan = 1.0)   

        self.terrain_type_idx = self.map.layers.index("terrain_type")
        self.map.data[self.terrain_type_idx].data = np.nan_to_num(self.map.data[self.terrain_type_idx].data, nan = 0.0)       
        
        self.update_torch_map()
        

    def update_torch_map(self):     
        self.geotraversableMap_torch = 1-torch.from_numpy(self.map.data[self.geo_traversability_idx].data).to(device=self.torch_device)                          
        self.terrainMap_torch        = torch.from_numpy(self.map.data[self.terrain_type_idx].data).to(device=self.torch_device)                                  
        self.elevationMap_torch      = torch.from_numpy(self.map.data[self.elevation_idx].data).to(device=self.torch_device)                    
        self.normal_vector_x_torch   = torch.from_numpy(self.normal_vector_x).to(device=self.torch_device)            
        self.normal_vector_y_torch   = torch.from_numpy(self.normal_vector_y).to(device=self.torch_device)            
        self.normal_vector_z_torch   = torch.from_numpy(self.normal_vector_z).to(device=self.torch_device)            
        

    def normalize_vector(self,vector):        
        return vector / torch.sqrt(torch.sum(vector**2,1)).view(-1,1).repeat(1,3)     

    def get_normal_vector_idx(self,idx):
        assert self.surface_normal_x_idx is not None, "surface normal x is not available"        
        if torch.sum(idx >= self.r_size*self.c_size) + torch.sum(idx < 0) > 0:      
            default_normal_vector = torch.zeros(len(idx),3)    
            default_normal_vector[:,2] = 1.0
            return default_normal_vector           
        else:
            normal_vector = torch.vstack((torch.index_select(self.normal_vector_x_torch,0,idx.squeeze()), 
                                            torch.index_select(self.normal_vector_y_torch,0,idx.squeeze()),
                                                torch.index_select(self.normal_vector_z_torch,0,idx.squeeze())))          
            normal_vector = torch.transpose(normal_vector,0,1)            
            return self.normalize_vector(normal_vector)

    def get_normal_vector(self,pose):
        if self.map is None or self.surface_normal_x_idx is None:
            return torch.Tensor([0,0,1])
        idx = self.pose2idx(pose)
        return self.get_normal_vector_idx(idx)                
    

    def get_terrain_type_idx(self,idx):
        if self.terrain_type_idx is None:
            return torch.zeros(len(idx))
        if torch.sum(idx >= self.r_size*self.c_size) > 0:
            print("idx out of bound")            
            return torch.zeros(len(idx))       
        return torch.index_select(self.terrainMap_torch,0,idx.squeeze())
        
       
    def get_terrain_type(self,pose):        
        idx = self.pose2idx(pose)
        if torch.sum(idx < 0) > 0:
            return torch.zeros(len(idx))
        return self.get_terrain_type_idx(idx)

    
    def get_elevation_idx(self,idx):
        if self.elevation_idx is None:
            return torch.zeros(len(idx))
        if torch.sum(idx >= self.r_size*self.c_size) > 0:
            print("idx out of bound")            
            return torch.zeros(len(idx))
        else: 
            return torch.index_select(self.elevationMap_torch,0,idx.squeeze())
           
    
    def get_elevation(self,pose):
        idx = self.pose2idx(pose)
        if torch.sum(idx < 0) > 0:
            return torch.zeros(len(idx))
            
        return self.get_elevation_idx(idx)        

  
    def get_rollpitch(self,pose):
        if not torch.is_tensor(pose):
            pose = torch.tensor(pose)  

        idx =self.pose2idx(pose)
        yaw = pose[:,2]
        default_rpy = torch.zeros(len(pose),3).to(device=self.torch_device) 
        default_rpy[:,2] = yaw
        if self.map is None:
            return default_rpy

        if torch.sum(idx) < 0:
            return default_rpy
        if self.surface_normal_x_idx is None:
            return default_rpy    
        normal_vector = self.get_normal_vector_idx(idx)             
        yaw_vec = torch.hstack([torch.cos(yaw).view(-1,1),torch.sin(yaw).view(-1,1),torch.zeros(len(yaw)).view(-1,1).to(device=self.torch_device)])
        yaw_vec[:,2] = -1*(normal_vector[:,0]*yaw_vec[:,0]+normal_vector[:,1]*yaw_vec[:,1])/(normal_vector[:,2]+1e-10)
        yaw_vec = self.normalize_vector(yaw_vec)                
        ry = torch.asin(yaw_vec[:,2])
        rz = torch.acos(yaw_vec[:,0] / ( torch.cos(ry)+1e-5))
        rx = torch.acos(normal_vector[:,2]/ (torch.cos(ry)+1e-5))
        roll = -1*rx
        pitch = -1*ry 
        yaw = rz 
        roll = wrap_to_pi_torch(roll)
        pitch = wrap_to_pi_torch(pitch)
        # assert roll is not np.nan, "idx is out of bound"                    
        return torch.hstack([roll.view(-1,1), pitch.view(-1,1), yaw.view(-1,1)])
        
        
    def pose2idx(self,pose):  
        grid_c_idx = ((self.right_corner_x - pose[:,0].view(-1,1)) / self.map_resolution).int()
        grid_r_idx = ((self.right_corner_y - pose[:,1].view(-1,1)) / self.map_resolution).int()
        if torch.sum(grid_c_idx >= self.c_size) > 0:
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device) 
        if torch.sum(grid_r_idx >= self.r_size) > 0:            
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device)         
        idx = grid_c_idx + grid_r_idx*self.r_size 
        if torch.sum(idx >= self.c_size*self.r_size) + torch.sum(idx < 0) > 0:
            return -1*torch.ones(len(grid_c_idx)).int().to(device=self.torch_device)                  
        return idx.int()


    def idx2grid(self,idx):
        r_idx  = (idx/self.c_size).int()
        c_idx  = (idx%self.c_size).int()
        return torch.transpose(torch.vstack((r_idx.squeeze(),c_idx.squeeze())),0,1)

    
    def get_sub_map_idx(self,idx,local_map_size):        
        map = self.geotraversableMap_torch        
        ## To do, currently the kernel size is manually fixed as 5.. 
        # need to implement auto generation of kernel given the local_map_size
        rc15 = map[idx.long()-2-self.c_size*2] 
        rc14 = map[idx.long()-1-self.c_size*2] 
        rc13 = map[idx.long()-self.c_size*2] 
        rc12 = map[idx.long()+1-self.c_size*2] 
        rc11 = map[idx.long()+2-self.c_size*2]        

        rc25 = map[idx.long()-2-self.c_size] 
        rc24 = map[idx.long()-1-self.c_size]
        rc23 = map[idx.long()-self.c_size] 
        rc22 = map[idx.long()+1-self.c_size]
        rc21 = map[idx.long()+2-self.c_size]

        rc35 = map[idx.long()-2] 
        rc34 = map[idx.long()-1]
        rc33 = map[idx.long()]
        rc32 = map[idx.long()+1] 
        rc31 = map[idx.long()+2]

        rc45 = map[idx.long()-2+self.c_size]
        rc44 = map[idx.long()-1+self.c_size]
        rc43 = map[idx.long()+self.c_size] 
        rc42 = map[idx.long()+1+self.c_size]
        rc41 = map[idx.long()+2+self.c_size] 

        rc55 = map[idx.long()-2+self.c_size*2]
        rc54 = map[idx.long()-1+self.c_size*2]
        rc53 = map[idx.long()+self.c_size*2]
        rc52 = map[idx.long()+1+self.c_size*2]
        rc51 = map[idx.long()+2+self.c_size*2]

        rc1 = torch.hstack([rc15,rc14,rc13,rc12,rc11])
        rc2 = torch.hstack([rc25,rc24,rc23,rc22,rc21])
        rc3 = torch.hstack([rc35,rc34,rc33,rc32,rc31])
        rc4 = torch.hstack([rc45,rc44,rc43,rc42,rc41])
        rc5 = torch.hstack([rc55,rc54,rc53,rc52,rc51])
        output_submaps = torch.stack([torch.transpose(rc1,0,1),torch.transpose(rc2,0,1),torch.transpose(rc3,0,1),torch.transpose(rc4,0,1),torch.transpose(rc5,0,1)])
        output_submaps = torch.permute(output_submaps,(2,0,1))
        
        return output_submaps


    def get_sub_map(self,pose,local_map_size):
        idx = self.pose2idx(pose)
        return self.get_sub_map_idx(idx,local_map_size)
        

    def compute_gpkernel_smoothing_cost(self,xmean,ymean,x_var,y_var, mahalanobis_distances):        
        mahalanobis_distances_torch = torch.transpose(torch.from_numpy(mahalanobis_distances).to(device = self.torch_device),0,1)
        dist_ = mahalanobis_distances_torch.reshape(-1,1).squeeze()
        pose = torch.hstack([xmean.view(-1,1),ymean.view(-1,1)])
        ## up to 99 % 
        xsig = torch.sqrt(x_var)         
        ysig = torch.sqrt(y_var)          
        xsig = torch.max(torch.hstack((xsig.view(-1,1),1e-5*torch.ones(len(xsig.view(-1,1))).to(device=self.torch_device).view(-1,1))),dim=1).values
        ysig =  torch.max(torch.hstack((ysig.view(-1,1),1e-5*torch.ones(len(ysig.view(-1,1))).to(device=self.torch_device).view(-1,1))),dim=1).values
        
        local_map_sizes = torch.tensor([self.kernel_size,self.kernel_size]).to(device= self.torch_device).int()
        submaps = self.get_sub_map(pose,local_map_sizes)        
        gpkernel = gaussianKN2D(local_map_sizes, rsig=ysig,csig=xsig)
        gpkernel_tensor = torch.stack(gpkernel,dim=0)
        submap_dot_gpkernel = torch.bmm(submaps,gpkernel_tensor)
        costs = torch.sum(submap_dot_gpkernel,dim=(1,2))
        ## costs => uncertain traversability cost smoothed by gpkernel 
        ## dist_ =>  mean error distribution distance from the nominal model to gp updated model 
        costs_with_scale = (costs*self.kernel_dist_cost_scale + dist_*self.prediction_diff_cost_scale).cpu().numpy()        
        costs_with_scale_numpy = np.array(costs_with_scale).reshape(xmean.shape[0],xmean.shape[1])
        
        return costs_with_scale_numpy

    
    def compute_rollover_cost(self,xpreds):
        ## we could assign infinite cost to the case when rollover index goes over the threshold (0.3)
        if not torch.is_tensor(xpreds):
            xpreds = torch.from_numpy(xpreds).to(device=self.torch_device)
        if xpreds.shape[0] < 2:
            return torch.zeros(len(xpreds)).to(device=self.torch_device)        
        
        rollover_costs = torch.zeros(xpreds.shape[1]).to(device=self.torch_device)        
        for i in range(xpreds.shape[0]-1):            
            accy = (xpreds[i+1,:,4]-xpreds[i,:,4])/self.dt
            rollover_idx = accy / (self.vehicle_model.width * self.vehicle_model.g) * (2*self.vehicle_model.h)
            rollover_apply_idx = rollover_idx > 0.3
            rollover_costs = rollover_costs+ rollover_idx*rollover_apply_idx            
            
        return rollover_costs 

    def compute_mahalanobis_distance(self,nominal_pose,sample_pose_mean,sample_pose_variance):        
        dists = np.zeros([nominal_pose.shape[0],nominal_pose.shape[1]])
        diff_vec = (nominal_pose - sample_pose_mean)        
        for i in range(nominal_pose.shape[1]):
            # i -> sample             
            for j in range(nominal_pose.shape[0]):
                # j-> time horizon 
                tmp_vec = diff_vec[j,i,:]
                tmp_var_diag = np.diag(sample_pose_variance[j,i,:])                
                dists[j][i] = np.sqrt(np.matmul(np.matmul(tmp_vec.T,tmp_var_diag),tmp_vec))        
        return dists
    
    def get_best_path(self,xmean,ymean,x_var,y_var,xpreds,goal=None):                  
        sample_pose_mean = np.transpose(np.stack([xmean,ymean]))
        sample_pose_variance = np.transpose(np.stack([x_var,y_var]))
        nominal_pose = xpreds[:,:,0:2]
        mahalanobis_distances = self.compute_mahalanobis_distance(nominal_pose,sample_pose_mean,sample_pose_variance)        
        xmean = torch.from_numpy(xmean).to(device = self.torch_device)
        ymean = torch.from_numpy(ymean).to(device = self.torch_device)
        x_var = torch.from_numpy(x_var).to(device = self.torch_device)
        y_var = torch.from_numpy(y_var).to(device = self.torch_device)        
        
        gpkernel_costs =  self.compute_gpkernel_smoothing_cost(xmean,ymean,x_var,y_var,mahalanobis_distances)                    
        gpkernel_costs_sum = np.sum(gpkernel_costs,axis = 1)                         
        normalized_gpkernel_costs_sum = gpkernel_costs_sum / (np.linalg.norm(gpkernel_costs_sum)+1e-5)                

        rollover_costs = self.compute_rollover_cost(xpreds).cpu().numpy()   
        normalized_rollover_costs =rollover_costs / (np.linalg.norm(rollover_costs) +1e-5)
        total_costs = normalized_gpkernel_costs_sum + normalized_rollover_costs*self.rollover_cost_scale
        
        if goal is not None:            
            goal_torch = torch.tensor(goal).to(device = self.torch_device)  
            dists = dist2d(torch.transpose(torch.vstack((xmean[:,-1],ymean[:,-1])),0,1),goal_torch.repeat(xmean[:,-1].shape[0],1)).cpu().numpy()
            dists_cost = dists
            # total_costs = total_costs + dists_cost     
            normalized_dist_cost = dists_cost / (np.linalg.norm(dists_cost)+1e-5)                                                       
            total_costs = total_costs +  normalized_dist_cost *self.dist_heuristic_cost_scale
        
        best_path_idx = np.argmin(total_costs)          
        best_path = xpreds[:,best_path_idx,:]
        
        return best_path, total_costs