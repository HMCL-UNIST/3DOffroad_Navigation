import numpy as np
import rospkg
import torch 
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch
import matplotlib.pyplot as plt
import time

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gplogger')


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3])),
            batch_shape=torch.Size([3])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )





class GPModel:
    def __init__(self,dt = 0.1, terrain_type = 0, data_file_name = "test_data.npz", model_file_name = "GP_129.pth",model_load = False):
        self.dt = dt
        data_dir = pkg_dir+"/data/"+data_file_name        
        self.model_file_name = pkg_dir+"/data/gp_model/"+model_file_name
        data = np.load(data_dir)
        xstate = data['xstate']
        xpredState = data['xpredState']
        self.gp_ready = False
        self.terrain_type = terrain_type
        
        # x, y, psi, vx, vy, wz, z roll, pitch , delta , accelx
        # 0  1  2    3  4   5   6   7,   8 ,      9 ,     10 

        #                                        x(2),     y(3),     psi(4),    vx(5),   vy(6),  omega(7), delta(8)
# add_state = np.array([u_pred[0,0],u_pred[1,0],xinit[0],xinit[1], xinit[2], xinit[3], xinit[4], xinit[5], xinit[6]])                    
        pred_states = xpredState[0:-1,[3,4,5]]
        true_states = xstate[1:,[3,4,5]]
        err_states = true_states - pred_states         
        self.y_train = err_states
        
        self.X_train = np.empty([len(err_states[:,0]), 7])        
        for i in range(len(err_states[:,0])):            
            self.X_train[i,:] = xstate[i,[3,4,5,7,8,9,10]] # x_train = [vx, vy, omega, roll, pitch, delta, accelx]
            
        self.X_train = torch.from_numpy(self.X_train).cuda().float()
        self.y_train_transpose = np.transpose(self.y_train)
        self.y_train_transpose = torch.from_numpy(self.y_train_transpose).cuda().float()
        
        self.y_train = torch.from_numpy(self.y_train).cuda().float()
        
        self.training_iterations = 200
        model_save_dir = pkg_dir+"/data/gp_model"
        self.model_name = model_save_dir+'/GP_'+str(self.terrain_type)+'.pth'        
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3,noise_prior=gpytorch.priors.NormalPrior(0, 1000)).cuda().float()
        self.model = BatchIndependentMultitaskGPModel(self.X_train, self.y_train, self.likelihood).cuda().float()

        if model_load:
            self.model_load()
        else:
            self.model_train()

        self.gp_ready = True


        


    def model_load(self):
        print(self.model_file_name)
        state_dict = torch.load(self.model_file_name)    
        print(self.model.state_dict())    
        self.model.load_state_dict(state_dict)
        print("~~~~~~~~~~~~~~")
        print(self.model.state_dict())        
        print(f"Model has been loaded from : {self.model_file_name}")
        self.model.eval()
        self.likelihood.eval()

    def model_train(self):     
        self.model.train()
        self.likelihood.train()
        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        loss_set = []
        
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train_transpose)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))            
            optimizer.step()
            step_lr_scheduler.step()
            loss_set.append(loss.item())

        self.model.eval()
        self.likelihood.eval()       
        
        ## Save trained data    
        state_dict = self.model.state_dict()
        for param_name, param in self.model.named_parameters():
            param_items = param.cpu().detach().numpy()
            print(f'Parameter name: {param_name:42} value = {param_items}')
        torch.save(self.model.state_dict(), self.model_name)        

        print(f"Model has been saved as : {self.model_name}")



    def get_mean_or_sample(self,X_test,is_sample = False):                
        if not torch.is_tensor(X_test):
          X_test = torch.from_numpy(X_test).cuda().float()
          if X_test.dim() < 2:
            X_test = X_test.expand(1,len(X_test))        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():     
            f_preds = self.model(X_test)                     
            mean = f_preds.mean.cpu().numpy()  
            mean = mean.squeeze() 
            f_var = f_preds.variance.cpu().numpy()
            f_var = f_var.squeeze()
            sample = np.random.normal(mean,np.sqrt(f_var))               
        if is_sample:
            return mean, sample
        else:
            return mean

        
        

    def gp_eval(self,X_test):
        if not torch.is_tensor(X_test):
          X_test = torch.from_numpy(X_test).cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():        
        # This contains predictions for both outcomes as a list        
            start_time = time.time()            
            predictions = self.likelihood(self.model(X_test))
            end_time = time.time()
            print("--- %s seconds ---" % (end_time - start_time))
            mean = predictions.mean.cpu()
            lower, upper = predictions.confidence_region()
        y_predicted_mean = mean.numpy()                
        
        return y_predicted_mean, upper, lower

    def draw_output(self,y_predicted_mean,lower,upper,ground_truth= None):
        with torch.no_grad():
            Xaxis = np.linspace(0,0.05*len(y_predicted_mean[:,0]),len(y_predicted_mean[:,0]))
            # Initialize plot
            f, ax = plt.subplots(3)

            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu()
            if torch.is_tensor(lower):
                lower = lower.cpu()
            if torch.is_tensor(upper):
                upper = upper.cpu()

            for i in range(len(ax)):
                if ground_truth is not None:
                    ax[i].plot(Xaxis, ground_truth[:,i], 'k*')
                # Plot predictive means as blue line
                ax[i].plot(Xaxis, y_predicted_mean[:,i], 'b')
                # Shade between the lower and upper confidence bounds
                ax[i].fill_between(Xaxis, lower[:,i], upper[:,i], alpha=0.5)                
                if ground_truth is not None:
                    ax[i].legend(['true', 'Mean', 'Confidence'])
                else:
                    ax[i].legend(['Mean', 'Confidence'])

                
            plt.show()
  
