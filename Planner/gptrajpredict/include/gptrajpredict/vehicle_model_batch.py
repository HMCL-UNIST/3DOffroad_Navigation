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
  @brief: compute the samples for predictive path distribution using 3D vehicle dynamics  
"""

from this import s
import numpy as np
from regex import P
import rospkg
import math
from gptrajpredict.utils_batch import b_to_g_rot
import torch


class VehicleModel:
    def __init__(self, dt = 0.1, N_node = 10, map_info = None, gp_n_sample = 10, gpmodels = None, gpmodel = None,device_ = "cuda", input_random = False):
        self.device = device_
        
        self.local_map = map_info
        self.N = N_node
        self.m = 25
        self.width = 0.45
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/math.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/math.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt
        self.max_delta = 25*math.pi/180.0        
        self.max_accel = 2.0        
        self.max_dccel = -2.0
        self.gpmodel = gpmodel        
        self.gp_n_sample =  gp_n_sample
        self.input_random = input_random

        ## sampled accel and delta for motion primitivies         
        accel_set = torch.linspace(self.max_dccel,self.max_accel,3).to(device=self.device)        
        delta_set = torch.linspace(-self.max_delta,self.max_delta,21).to(device=self.device)        
        self.uset = torch.hstack([torch.meshgrid(delta_set,accel_set)[0].reshape(-1,1),torch.meshgrid(delta_set,accel_set)[1].reshape(-1,1)])
        self.usets = self.uset.repeat(self.gp_n_sample,1)
        ###### Add randomness to input sets #####
        if self.input_random:
            random_add = torch.zeros(self.usets.shape).to(device=self.device)
            # random_add = self.usets +torch.rand(self.usets.shape).to(device=self.device)
            random_add[:,0] = self.usets[:,0] + (torch.rand(self.usets[:,0].shape).to(device=self.device)-0.5)*self.max_delta
            random_add[:,1] = self.usets[:,1] + (torch.rand(self.usets[:,1].shape).to(device=self.device)-0.5)*self.max_accel            
            random_add[:,0] = torch.clamp(random_add[:,0],-1*self.max_delta,self.max_delta)
            random_add[:,1] = torch.clamp(random_add[:,1],self.max_dccel,self.max_accel)
            self.usets = random_add
        #########################################
        self.usets_backup = torch.tensor(self.usets).clone()
        if gpmodels is not None:
            self.gpmodels = gpmodels
            self.n_gp_model = len(self.gpmodels)        
        self.B = len(accel_set)*len(delta_set)

    def reset_input(self):
        self.usets = self.uset.repeat(self.gp_n_sample,1)
        ###### Add randomness to input sets #####
        if self.input_random:
            random_add = torch.zeros(self.usets.shape).to(device=self.device)            
            random_add[:,0] = self.usets[:,0] + (torch.rand(self.usets[:,0].shape).to(device=self.device)-0.5)*self.max_delta
            random_add[:,1] = self.usets[:,1] + (torch.rand(self.usets[:,1].shape).to(device=self.device)-0.5)*self.max_accel            
            random_add[:,0] = torch.clamp(random_add[:,0],-1*self.max_delta,self.max_delta)
            random_add[:,1] = torch.clamp(random_add[:,1],self.max_dccel,self.max_accel)
            self.usets = random_add
        #########################################

    def compute_slip(self, x,u):   
        clip_vx = torch.max(torch.hstack((x[:,3].view(-1,1),torch.ones(len(x[:,3])).to(device=self.device).view(-1,1))),dim =1).values   
        alpha_f = u[:,0] - (x[:,4]+self.Lf*x[:,5])/clip_vx
        alpha_r = (-x[:,4]+self.Lr*x[:,5])/clip_vx        
        return alpha_f, alpha_r
      
    def compute_normal_force(self,x,u,roll,pitch):                  
        Fzf = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L + self.h*self.m/self.L*(u[:,1]+self.g*torch.sin(pitch))
        Fzr = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L - self.h*self.m/self.L*(u[:,1]+self.g*torch.sin(pitch))
        return Fzf, Fzr
    
    def batch_predict_multistep_multiinput_with_sample_gp(self,x,goal = None):            
        dist_to_goal = 0.0
        if goal is not None:            
            dist_to_goal = np.hypot(x[0]-goal[0], x[1]-goal[1])
        
        n_sample = self.gp_n_sample
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.device)               
        ## Limit the initial velocity state 
        x[3] = torch.clip(x[3],0.0, 1.0)
        print("dist to goal = " + str(dist_to_goal))
        if dist_to_goal >= 8.0:            
            if x[3] <=0.1:
                self.usets[:,1] = torch.clip(self.usets[:,1],self.max_accel,1e3)
            else:
                self.usets = self.usets_backup
            
        else:            
            self.usets = self.usets_backup.clone()
            print("backup contrl")

        x[3]  = torch.max(torch.stack((x[3],torch.zeros(1).squeeze().to(device=self.device))))
        x_ = x.view(1, -1).repeat(self.B, 1)                
        x_set = x_.repeat(n_sample,1,1)  #  [ number_of_sample, number_of_batch, number_of_states]        
        predicted_x_set = torch.tile(torch.clone(x_set),(self.N,1,1,1))
        predicted_x_set_mean = torch.clone(predicted_x_set)
        predicted_x_set_nominal = torch.clone(predicted_x_set)
        
        ####
        # N is the number of step to predict         
        tmp_x_set = x_set.view(-1,x_set.shape[-1])        
        nx_tmp = torch.clone(tmp_x_set)
        # same for GP mean dynamics 
        tmp_x_set_mean = torch.clone(tmp_x_set)
        nx_tmp_mean = torch.clone(nx_tmp)
        # same for nominal dynamics 
        tmp_x_set_nominal = torch.clone(tmp_x_set)
        nx_tmp_nominal = torch.clone(nx_tmp)


        ## ## Update for each prediction step
        for i in range(self.N):                
            ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
            ########################## GP sample dynamics update  ###########################            
            # ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ###
            gp_input = tmp_x_set[:,[3,4,5,7,8]]                      
            terrain_types = self.local_map.get_terrain_type(tmp_x_set[:,[0,1]]).view(-1,1)      
            # output dimension is 3 
            gpmean = torch.zeros(tmp_x_set.shape[0],3).to(device=self.device)
            gpoutput = torch.zeros(tmp_x_set.shape[0],3).to(device=self.device)            
            ## Gp models update for  Different types of terrain 
            for l in range(self.n_gp_model):                                                                 
                indexset = (terrain_types.squeeze() == self.gpmodels[l].terrain_type).nonzero(as_tuple = False).long().squeeze()                
                if indexset.dim() != 0 and len(indexset)==0:
                    continue                
                gp_input_in_one_row = gp_input    
                filtered_gp_intput_in_one_row =gp_input_in_one_row[indexset,:]                             
                inputs_tmp = self.usets[indexset,:]
                gpinput_states_tmp = torch.hstack([filtered_gp_intput_in_one_row,inputs_tmp])                
                gpmean_tmp, gpoutput_tmp = self.gpmodels[l].get_mean_or_sample(gpinput_states_tmp, is_sample = True)                            
                gpmean[indexset,:] = gpmean_tmp
                gpoutput[indexset,:] = gpoutput_tmp            
            usets = self.usets           
                               
            ######## get roll and pitch from Map ######## 
            if self.local_map is not None:            
                pose = torch.transpose(torch.vstack([tmp_x_set[:,0],tmp_x_set[:,1],tmp_x_set[:,2]]),0,1)
                rpy_tmp = self.local_map.get_rollpitch(pose)               
                roll = rpy_tmp[:,0]
                pitch = rpy_tmp[:,1]
                tmp_x_set[:,6] = self.local_map.get_elevation(pose)                
                nx_tmp[:,6] = torch.where(tmp_x_set[:,6] ==0, nx_tmp[:,6],tmp_x_set[:,6])
            else:            
                roll = tmp_x_set[:,7]
                pitch = tmp_x_set[:,8]
            
            delta = usets[:,0]
            axb = usets[:,1]
            rot_base_to_world = b_to_g_rot(roll,pitch,tmp_x_set[:,2]).double()
            Fzf, Fzr = self.compute_normal_force(tmp_x_set,usets,roll,pitch)
            alpha_f, alpha_r = self.compute_slip(tmp_x_set,usets)
            Fyf = Fzf * alpha_f            
            Fyr =  Fzr * alpha_r            
            local_vel = torch.hstack([tmp_x_set[:,3].view(-1,1),tmp_x_set[:,4].view(-1,1),torch.zeros(len(tmp_x_set[:,3])).to(device=self.device).view(-1,1)]).view(-1,3,1).double()            
            vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
            
            vxw = vel_in_world[:,0]
            vyw = vel_in_world[:,1]
            vzw = vel_in_world[:,2]   
                        
            nx_tmp[:,0] = nx_tmp[:,0]+self.dt*vxw
            nx_tmp[:,1] = nx_tmp[:,1]+self.dt*vyw
            nx_tmp[:,2] = nx_tmp[:,2]+self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*tmp_x_set[:,5])
            nx_tmp[:,3] = nx_tmp[:,3]+self.dt*axb + gpoutput[:,0]
            nx_tmp[:,3] = torch.max(torch.hstack((nx_tmp[:,3].view(-1,1),torch.zeros(len(tmp_x_set[:,3])).to(device=self.device).view(-1,1))),dim =1).values           
            nx_tmp[:,4] = nx_tmp[:,4]+self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-tmp_x_set[:,3]*tmp_x_set[:,5])+gpoutput[:,1]
            nx_tmp[:,5] = nx_tmp[:,5]+self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz) +gpoutput[:,2]
            nx_tmp[:,6] = nx_tmp[:,6] +self.dt*vzw

            predicted_x_set[i,:,:,:] = nx_tmp.view(n_sample,self.B,-1)
            tmp_x_set = torch.clone(nx_tmp)                        
            
            ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
            ########################## Mean dynamics update  ################################            
            # ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ###
            
            gp_input_mean = tmp_x_set_mean[:,[3,4,5,7,8]]                      
            terrain_types = self.local_map.get_terrain_type(tmp_x_set_mean[:,[0,1]]).view(-1,1)            
            # output dimension is 3 
            gpmean_mean = torch.zeros(tmp_x_set_mean.shape[0],3).to(device=self.device)                        
            ## Gp models update for  Different types of terrain 
            for l in range(self.n_gp_model):                                                                 
                indexset = (terrain_types.squeeze() == self.gpmodels[l].terrain_type).nonzero(as_tuple = False).long().squeeze()                                
                if indexset.dim() != 0 and len(indexset)==0:
                    continue                
                gp_input_in_one_row = gp_input_mean   
                filtered_gp_intput_in_one_row =gp_input_in_one_row[indexset,:]                             
                inputs_tmp = self.usets[indexset,:]
                gpinput_states_tmp = torch.hstack([filtered_gp_intput_in_one_row,inputs_tmp])
                gpmean_tmp = self.gpmodels[l].get_mean_or_sample(gpinput_states_tmp, is_sample = False)                                         
                gpmean_mean[indexset,:] = gpmean_tmp             
              
            if self.local_map is not None:            
                pose = torch.transpose(torch.vstack([tmp_x_set_mean[:,0],tmp_x_set_mean[:,1],tmp_x_set_mean[:,2]]),0,1)
                rpy_tmp = self.local_map.get_rollpitch(pose)               
                roll = rpy_tmp[:,0]
                pitch = rpy_tmp[:,1]
                tmp_x_set_mean[:,6] = self.local_map.get_elevation(pose)                
                nx_tmp_mean[:,6] = torch.where(tmp_x_set_mean[:,6] ==0, nx_tmp_mean[:,6],tmp_x_set_mean[:,6])
            else:                  
                roll = tmp_x_set_mean[:,7]
                pitch = tmp_x_set_mean[:,8]                

            delta = usets[:,0]
            axb = usets[:,1]
            rot_base_to_world = b_to_g_rot(roll,pitch,tmp_x_set_mean[:,2]).double()
            Fzf, Fzr = self.compute_normal_force(tmp_x_set_mean,usets,roll,pitch)
            alpha_f, alpha_r = self.compute_slip(tmp_x_set_mean,usets)
            Fyf = Fzf * alpha_f            
            Fyr =  Fzr * alpha_r            
            local_vel = torch.hstack([tmp_x_set_mean[:,3].view(-1,1),tmp_x_set_mean[:,4].view(-1,1),torch.zeros(len(tmp_x_set_mean[:,3])).to(device=self.device).view(-1,1)]).view(-1,3,1).double()            
            vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
            
            vxw = vel_in_world[:,0]
            vyw = vel_in_world[:,1]
            vzw = vel_in_world[:,2] 
            
            nx_tmp_mean[:,0] = nx_tmp_mean[:,0]+self.dt*vxw
            nx_tmp_mean[:,1] = nx_tmp_mean[:,1]+self.dt*vyw
            nx_tmp_mean[:,2] = nx_tmp_mean[:,2]+self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*tmp_x_set_mean[:,5])
            nx_tmp_mean[:,3] = nx_tmp_mean[:,3]+self.dt*axb +gpmean_mean[:,0]
            nx_tmp_mean[:,3] = torch.max(torch.hstack((nx_tmp_mean[:,3].view(-1,1),torch.zeros(len(tmp_x_set_mean[:,3])).to(device=self.device).view(-1,1))),dim =1).values           
            nx_tmp_mean[:,4] = nx_tmp_mean[:,4]+self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-tmp_x_set_mean[:,3]*tmp_x_set_mean[:,5]) +gpmean_mean[:,1]
            nx_tmp_mean[:,5] = nx_tmp_mean[:,5]+self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz) +gpmean_mean[:,2]
            nx_tmp_mean[:,6] = nx_tmp_mean[:,6] +self.dt*vzw
            predicted_x_set_mean[i,:,:,:] = nx_tmp_mean.view(n_sample,self.B,-1)
            tmp_x_set_mean = torch.clone(nx_tmp_mean)    
            
            ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
            ########################## Nominal dynamics update  ################################            
            # ##  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ###
            
            if self.local_map is not None:            
                pose = torch.transpose(torch.vstack([tmp_x_set_nominal[:,0],tmp_x_set_nominal[:,1],tmp_x_set_nominal[:,2]]),0,1)
                rpy_tmp = self.local_map.get_rollpitch(pose)               
                roll = rpy_tmp[:,0]
                pitch = rpy_tmp[:,1]
                tmp_x_set_nominal[:,6] = self.local_map.get_elevation(pose)                
                nx_tmp_nominal[:,6] = torch.where(tmp_x_set_nominal[:,6] ==0, nx_tmp_nominal[:,6],tmp_x_set_nominal[:,6])
            else:                  
                roll = tmp_x_set_nominal[:,7]
                pitch = tmp_x_set_nominal[:,8]                
            
            delta = usets[:,0]
            axb = usets[:,1]
            rot_base_to_world = b_to_g_rot(roll,pitch,tmp_x_set_nominal[:,2]).double()
            Fzf, Fzr = self.compute_normal_force(tmp_x_set_nominal,usets,roll,pitch)
            alpha_f, alpha_r = self.compute_slip(tmp_x_set_nominal,usets)
            Fyf = Fzf * alpha_f            
            Fyr =  Fzr * alpha_r            
            local_vel = torch.hstack([tmp_x_set_nominal[:,3].view(-1,1),tmp_x_set_nominal[:,4].view(-1,1),torch.zeros(len(tmp_x_set_nominal[:,3])).to(device=self.device).view(-1,1)]).view(-1,3,1).double()            
            vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
            
            vxw = vel_in_world[:,0]
            vyw = vel_in_world[:,1]
            vzw = vel_in_world[:,2] 

            nx_tmp_nominal[:,0] = nx_tmp_nominal[:,0]+self.dt*vxw
            nx_tmp_nominal[:,1] = nx_tmp_nominal[:,1]+self.dt*vyw
            nx_tmp_nominal[:,2] = nx_tmp_nominal[:,2]+self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*tmp_x_set_nominal[:,5])
            nx_tmp_nominal[:,3] = nx_tmp_nominal[:,3]+self.dt*axb 
            nx_tmp_nominal[:,3] = torch.max(torch.hstack((nx_tmp_nominal[:,3].view(-1,1),torch.zeros(len(tmp_x_set_nominal[:,3])).to(device=self.device).view(-1,1))),dim =1).values           
            nx_tmp_nominal[:,4] = nx_tmp_nominal[:,4]+self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-tmp_x_set_nominal[:,3]*tmp_x_set_nominal[:,5]) 
            nx_tmp_nominal[:,5] = nx_tmp_nominal[:,5]+self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz) 
            nx_tmp_nominal[:,6] = nx_tmp_nominal[:,6] +self.dt*vzw
            predicted_x_set_nominal[i,:,:,:] = nx_tmp_nominal.view(n_sample,self.B,-1)
            tmp_x_set_nominal = torch.clone(nx_tmp_nominal)             

    
        return predicted_x_set.cpu().numpy(), predicted_x_set_mean.cpu().numpy(), predicted_x_set_nominal.cpu().numpy()
        
    