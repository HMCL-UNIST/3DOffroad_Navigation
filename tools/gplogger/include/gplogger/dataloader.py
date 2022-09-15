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
  @brief: Dataloader
  @details: store state information
"""
import numpy as np

class DataLoader:
    def __init__(self, input_dim = 2, state_dim = 9, dt = 0.1, terrain_id = 0):                               
        self.Xstates = np.empty((0,input_dim+state_dim))
        self.XpredStates = np.empty((0,input_dim+state_dim))
        self.dt = dt
        self.n_data_set = 0
        self.terrain_type = terrain_id
        
    def file_load(self,file_path):
        data = np.load(file_path)
        self.Xstates = data['xstate']
        self.XpredStates = data['xpredState']
        
    def file_save(self,fil_dir):   
        np.savez(fil_dir,xstate = self.Xstates, xpredState = self.XpredStates)

    def append_state(self,xstate_,XpredStates_):
        self.Xstates = np.append(self.Xstates,[xstate_],axis = 0)        
        self.XpredStates = np.append(self.XpredStates,[XpredStates_],axis = 0)       
        self.n_data_set= self.n_data_set+1 

    def append_state_for_traj(self,xstate_):
        self.Xstates = np.append(self.Xstates,[xstate_],axis = 0)                
        self.n_data_set= self.n_data_set+1 
    
