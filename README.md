# Learning-based Uncertainty-aware 3D Offroad Navigation

**A safe, efficient, and agile ground vehicle navigation algorithm for 3D off-road terrain environments. A video of the demonstration of the method can be found on [YouTube](https://www.youtube.com/watch).**



<p align='center'>
    <img src="/jpg/gpmppi3.gif" alt="drawing" width="400"/>
    <img src="/jpg/hojin1.jpg" alt="drawing" width="280"/>
</p>




## System architecture
<p align='center'>
    <img src="/jpg/hojin2.jpg" alt="drawing" width="1000"/>
</p>


We design a system that learns the terrain-induced uncertainties from driving data and encodes the learned uncertainty distribution into the
traversability cost for path evaluation. The navigation path is then designed to optimize the uncertainty-aware traversability cost, resulting in a safe and agile vehicle maneuver.  

## Dependency

Tested with ROS Melodic. 

1. To install Gazebo simulation environment
->  Please follow the installation tutorial in "https://github.com/AutoRally/autorally.git" 
2. For Mapping modules 
-> Dependencis can be found at "https://github.com/leggedrobotics/traversability_estimation.git"
3. NVIDIA Graphic card and Gpytorch is required 
```
pip install gpytorch
```
4. pytorch is required 
-> https://pytorch.org/get-started/locally/



## Install

Use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/HMCL-UNIST/offroad_navigation.git 
cd ..
catkin build 
```

## Record vehicle state data for GP model 
Record state history for GP model. 
  - **Joystick required **. 
  ros joystick input is used to control the vehicle manually while recording the state messages. 
  To begin the recording process, first RUN 
  ```
    roslaunch autorally_gazebo gpmppi3d.launch
    roslaunch gplogger gplogger.launch
    roslaunch lowlevel_ctrl lowlevel_ctrl.launch
  ```

*To enable data logging*
1. publish topic  /gp_data_logging as "true" 

*To save data *
2. publish topic  /gp_data_save as "true" 


## Run the package

1. Run the Gazebo Simulation :
```
roslaunch autorally_gazebo gpmppi3d.launch
```

2. Run the preprocessing Modules  :
```
roslaunch elevation_mapping elevation_mapping.launch
roslaunch traversability_estimation traversability_estimation.launch
```

3. Run the uncertainty-aware path planning Module  :
```
roslaunch gptrajpredict gptrajpredict.launch
```

4. Run the Model predictive controller (MPPI) :
```
roslaunch mppi_ctrl mppi_ctrl.launch
```

5. Run low level controller 
```
roslaunch lowlevel_ctrl lowlevel_ctrl.launch
```



## Paper 
Hojin Lee, Junsung Kwon, and Cheolhyeon Kwon, Learning-based Uncertainty-aware Navigation in 3D Off-Road Terrains, 2023 ICRA, under review 


## Acknowledgement
 **I would like to express my sincere thanks to following**

- Our 3D Simulation environment and the Gazebo vehicle model is based on Autorally research platform  

```
(Goldfain, Brian, et al. "Autorally: An open platform for aggressive autonomous driving." IEEE Control Systems Magazine 39.1 (2019): 26-55.)  
```

- Elevation and traversability mapping modules used in preprocessing step are based on these awesome work. 
```
( P. Fankhauser, M. Bloesch, C. Gehring, M. Hutter, and R. Siegwart,
        “Robot-centric elevation mapping with uncertainty estimates,” in Mobile
          Service Robotics. World Scientific, 2014, pp. 433–440.) 
```       

```
(P. Fankhauser, M. Bloesch, and M. Hutter, “Probabilistic terrain
mapping for mobile robots with uncertain localization,” IEEE Robotics
and Automation Letters, vol. 3, no. 4, pp. 3019–3026, 2018.) 
```


``` 
(T. H. Y. Leung, D. Ignatyev, and A. Zolotas, “Hybrid terrain
traversability analysis in off-road environments,” in 2022 8th International
Conference on Automation, Robotics and Applications (ICARA).
IEEE, 2022, pp. 50–56.)
```
