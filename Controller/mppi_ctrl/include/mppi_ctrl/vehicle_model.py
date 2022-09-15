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
  @brief: 3D Vehicle model using torch
"""

from this import s
from regex import P
import rospkg
import math
from mppi_ctrl.utils import b_to_g_rot
import torch
from mppi_ctrl.mppi import MPPI

class VehicleModel:
    def __init__(self, device_ = "cuda", dt = 0.1, N_node = 10, map_info = None, mppi_n_sample = 10):
        self.local_map = map_info
        self.N = N_node
        self.m = 25
        self.width = 0.45
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/torch.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt 
        self.delta_rate_max = 50*math.pi/180.0 # rad/s
        self.rollover_cost_scale = 10.0        
        self.Q = torch.tensor([0.5, 0.5, 0.5])   
        self.mppi_n_sample = mppi_n_sample
        self.torch_device = device_
        self.state_dim = 7
        self.input_dim = 2
        self.lambda_ = 10.0
        self.horizon = self.dt* N_node
        self.noise_sigma = torch.diag(torch.ones(self.input_dim)*10,0)
        # delta and ax
        self.u_min = torch.tensor([-25*torch.pi/180.0, -3.0]).to(device= self.torch_device)
        self.u_max = self.u_min*-1

        self.mppi = MPPI(self.dynamics_update, self.running_cost, self.state_dim, self.noise_sigma, num_samples=self.mppi_n_sample, horizon=self.N,
                         lambda_=self.lambda_, u_min = self.u_min, u_max = self.u_max)
        self.target_path = None

    def set_path(self,path):
         # self.target_path --> [(x, y, psi), horizon(t)]
         self.target_path = path.repeat(self.mppi_n_sample,1,1)
         

    def compute_slip(self, x,u):   
        clip_vx = torch.max(torch.hstack((x[:,3].view(-1,1),torch.ones(len(x[:,3])).to(device=self.torch_device).view(-1,1))),dim =1).values   
        alpha_f = u[:,0] - (x[:,4]+self.Lf*x[:,5])/clip_vx
        alpha_r = (-x[:,4]+self.Lr*x[:,5])/clip_vx        
        return alpha_f, alpha_r
      
    def wrap_to_pi_torch(self,array):
        array[array<-torch.pi] = array[array<-torch.pi] + 2*torch.pi
        array[array>torch.pi] = array[array>torch.pi] - 2*torch.pi
        return array 


    def compute_normal_force(self,x,u,roll,pitch):          
        Fzf = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L + self.h*self.m/self.L*(u[:,1]+self.g*torch.sin(pitch))
        Fzr = self.Lr*self.m*self.g*torch.cos(pitch)*torch.cos(roll)/self.L - self.h*self.m/self.L*(u[:,1]+self.g*torch.sin(pitch))
        return Fzf, Fzr
     
    def running_cost(self, state, action, t,prev_state):
        # self.target_path --> [x, y, psi, horizon(t)]
        cost = torch.zeros(state.shape[0]).to(device=self.torch_device)
        state[:,2] = self.wrap_to_pi_torch(state[:,2])        
        self.target_path[:,2,t] = self.wrap_to_pi_torch(self.target_path[:,2,t])        

        ############################################################
        ################# Trajectory following cost ################
        ############################################################
        if self.target_path is not None:
            cost = self.Q[0]*((state[:,0]-self.target_path[:,0,t]))**2 +self.Q[1]*((state[:,1]-self.target_path[:,1,t]))**2 +self.Q[2]*((state[:,2]-self.target_path[:,2,t]))**2            

        ############################################################
        ################# roll over prevention cost ################
        ############################################################
        # predictedState = x, y, psi, vx, vy, wz, z, roll, pitch    
        accy = (state[:,4]-prev_state[:,4])/self.dt
        rollover_idx = accy / (self.width * self.g) * (2*self.h)
        rollover_idx = torch.where(rollover_idx < 0.3, 0.0,rollover_idx)
        rollover_cost_tmp = rollover_idx*self.rollover_cost_scale
        cost += rollover_cost_tmp

        return cost
        
    def dynamics_update(self,x,u):     
        # x(0), y(1), psi(2), vx(3), vy(4), wz(5) z(6) roll(7) pitch(8)                  
        # u(0) = delta, u(1) = ax 
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.torch_device)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.torch_device)    
        nx = torch.clone(x).to(device=self.torch_device)  
        ######## get roll and pitch from Map ######## 
        if self.local_map is not None:            
            pose = torch.transpose(torch.vstack([x[:,0],x[:,1],x[:,2]]),0,1)
            rpy_tmp = self.local_map.get_rollpitch(pose)               
            roll = rpy_tmp[:,0]
            pitch = rpy_tmp[:,1]
            y_ = rpy_tmp[:,2]            
            x[:,6] = self.local_map.get_elevation(pose)
            nx[:,6] = x[:,6]
        else:        
            roll = x[:,7]
            pitch = x[:,8]
            
        delta = u[:,0]
        axb = u[:,1]
        rot_base_to_world = b_to_g_rot(roll,pitch,x[:,2]).double()
        Fzf, Fzr = self.compute_normal_force(x,u,roll,pitch)
        alpha_f, alpha_r = self.compute_slip(x,u)
        Fyf = Fzf * alpha_f            
        Fyr =  Fzr * alpha_r        
        local_vel = torch.hstack([x[:,3].view(-1,1),x[:,4].view(-1,1),torch.zeros(len(x[:,3])).to(device=self.torch_device).view(-1,1)]).view(-1,3,1).double()        
        vel_in_world = torch.bmm(rot_base_to_world, local_vel).view(-1,3)
        
        vxw = vel_in_world[:,0]
        vyw = vel_in_world[:,1]
        vzw = vel_in_world[:,2]   
                    
        nx[:,0] = nx[:,0]+self.dt*vxw
        nx[:,1] = nx[:,1]+self.dt*vyw
        nx[:,2] = nx[:,2]+self.dt*(torch.cos(roll)/(torch.cos(pitch)+1e-10)*x[:,5])
        nx[:,3] = nx[:,3]+self.dt*axb 
        nx[:,3] = torch.max(torch.hstack((nx[:,3].view(-1,1),torch.zeros(len(x[:,3])).to(device=self.torch_device).view(-1,1))),dim =1).values           
        nx[:,4] = nx[:,4]+self.dt*((Fyf+Fyr+self.m*self.g*torch.cos(pitch)*torch.sin(roll))/self.m-x[:,3]*x[:,5])
        nx[:,5] = nx[:,5]+self.dt*((Fyf*self.Lf*torch.cos(delta)-self.Lr*Fyr)/self.Izz)
        nx[:,6] = nx[:,6]+self.dt*vzw
        
        return nx

