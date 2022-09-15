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
  @brief: VehicleModel
  @details: 3D vehicle model
"""
import numpy as np
from regex import P
import math
from gplogger.utils import b_to_g_rot


class VehicleModel:
    def __init__(self, dt = 0.1, map_info = None, gpmodel = None):
        self.local_map = map_info
        self.m = 25
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/math.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/math.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt
        self.gpmodel = gpmodel
        self.n_gpPredict_sample = 10        
       
    def compute_slip(self, x,u):
        clip_vx = max(1,x[3])
        alpha_f = u[0] - (x[4]+self.Lf*x[5])/clip_vx
        alpha_r = (-x[4]+self.Lr*x[5])/clip_vx
        return alpha_f, alpha_r
      
    def compute_normal_force(self,x,u,roll,pitch):        
        Fzf = self.Lr*self.m*self.g*math.cos(pitch)*math.cos(roll)/self.L + self.h*self.m/self.L*(u[1]+self.g*math.sin(pitch))
        Fzr = self.Lr*self.m*self.g*math.cos(pitch)*math.cos(roll)/self.L - self.h*self.m/self.L*(u[1]+self.g*math.sin(pitch))
        return Fzf, Fzr
    
    
   
    
    def dynamics_update(self,x,u,gp_enable = False, sample_update = False, N = 1):     
        # x(0), y(1), psi(2), vx(3), vy(4), wz(5) z(6) roll(7) pitch(8)                  
        # u(0) = delta, u(1) = ax 
        nx = x.copy()
        ######## get roll and pitch from Map ######## 
        if self.local_map is not None:
            pose = np.array([x[0], x[1], x[2]])                 
            roll,pitch,y_ = self.local_map.get_rollpitch(pose)                                    
            x[6] = self.local_map.get_elevation(pose)
            nx[6] = x[6]
        else:        
            roll = x[7]
            pitch = x[8]
            
        delta = u[0]
        axb = u[1]
        rot_base_to_world = b_to_g_rot(roll,pitch,x[2])
        Fzf, Fzr = self.compute_normal_force(x,u,roll,pitch)
        alpha_f, alpha_r = self.compute_slip(x,u)
        Fyf = Fzf * alpha_f            
        Fyr =  Fzr * alpha_r        
        vel_in_world = np.dot(rot_base_to_world,[x[3],x[4],0])
        vxw = vel_in_world[0]
        vyw = vel_in_world[1]
        vzw = vel_in_world[2]   
        
        if gp_enable: 
            # gpinput => # [vx, vy, omega, roll, pitch, delta, accelx]
            gpinput =  np.array([nx[3],nx[4],nx[5],nx[7],nx[8],delta,axb])
            # gpoutput => vx_error, vy_error, omega_z_error 
            if sample_update:
                gpoutput = self.gpmodel.get_mean_or_sample(gpinput, is_sample = True)
            else:
                gpoutput = self.gpmodel.get_mean_or_sample(gpinput, is_sample = False)
            
            # x(0), y(1), psi(2), vx(3), vy(4), wz(5) z(6)              
            nx[0] = nx[0]+self.dt*vxw
            nx[1] = nx[1]+self.dt*vyw
            nx[2] = nx[2]+self.dt*(math.cos(roll)/(math.cos(pitch)+1e-10)*x[5])
            nx[3] = nx[3]+self.dt*axb + gpoutput[0]
            nx[4] = nx[4]+self.dt*((Fyf+Fyr+self.m*self.g*math.cos(pitch)*math.sin(roll))/self.m-x[3]*x[5]) + gpoutput[1]
            nx[5] = nx[5]+self.dt*((Fyf*self.Lf*math.cos(delta)-self.Lr*Fyr)/self.Izz) + gpoutput[2]
            nx[6] = nx[6]+self.dt*vzw
        else:                       
            nx[0] = nx[0]+self.dt*vxw
            nx[1] = nx[1]+self.dt*vyw
            nx[2] = nx[2]+self.dt*(math.cos(roll)/(math.cos(pitch)+1e-10)*x[5])
            nx[3] = nx[3]+self.dt*axb 
            nx[4] = nx[4]+self.dt*((Fyf+Fyr+self.m*self.g*math.cos(pitch)*math.sin(roll))/self.m-x[3]*x[5])
            nx[5] = nx[5]+self.dt*((Fyf*self.Lf*math.cos(delta)-self.Lr*Fyr)/self.Izz)
            nx[6] = nx[6]+self.dt*vzw
            


        return nx


