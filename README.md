# offroad_navigation
learning based uncertainty-aware 3D offroad navigation


# LIO-SAM

**A real-time lidar-inertial odometry package. We strongly recommend the users read this document thoroughly and test the package with the provided dataset first. A video of the demonstration of the method can be found on [YouTube](https://www.youtube.com/watch?v=A0H8CoORZJU).**

<p align='center'>
    <img src="./config/doc/demo.gif" alt="drawing" width="800"/>
</p>

<p align='center'>
    <img src="./config/doc/device-hand-2.png" alt="drawing" width="200"/>
    <img src="./config/doc/device-hand.png" alt="drawing" width="200"/>
    <img src="./config/doc/device-jackal.png" alt="drawing" width="200"/>
    <img src="./config/doc/device-livox-horizon.png" alt="drawing" width="200"/>
</p>

## Menu

  - [**System architecture**](#system-architecture)

  - [**Package dependency**](#dependency)

  - [**Package install**](#install)

  - [**Prepare lidar data**](#prepare-lidar-data) (must read)

  - [**Prepare IMU data**](#prepare-imu-data) (must read)

  - [**Sample datasets**](#sample-datasets)

  - [**Run the package**](#run-the-package)

  - [**Other notes**](#other-notes)

  - [**Issues**](#issues)

  - [**Paper**](#paper)

  - [**TODO**](#todo)

  - [**Related Package**](#related-package)

  - [**Acknowledgement**](#acknowledgement)

## System architecture

<p align='center'>
    <img src="./config/doc/system.png" alt="drawing" width="800"/>
</p>

We design a system that maintains two graphs and runs up to 10x faster than real-time.
  - The factor graph in "mapOptimization.cpp" optimizes lidar odometry factor and GPS factor. This factor graph is maintained consistently throughout the whole test.
  - The factor graph in "imuPreintegration.cpp" optimizes IMU and lidar odometry factor and estimates IMU bias. This factor graph is reset periodically and guarantees real-time odometry estimation at IMU frequency.

## Dependency

This is the original ROS1 implementation of LIO-SAM. For a ROS2 implementation see branch `ros2`.

- [ROS](http://wiki.ros.org/ROS/Installation) (tested with Kinetic and Melodic. Refer to [#206](https://github.com/TixiaoShan/LIO-SAM/issues/206) for Noetic)
  ```
  sudo apt-get install -y ros-kinetic-navigation
  sudo apt-get install -y ros-kinetic-robot-localization
  sudo apt-get install -y ros-kinetic-robot-state-publisher
  ```
- [gtsam](https://gtsam.org/get_started/) (Georgia Tech Smoothing and Mapping library)
  ```
  sudo add-apt-repository ppa:borglab/gtsam-release-4.0
  sudo apt install libgtsam-dev libgtsam-unstable-dev
  ```

## Install

Use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/TixiaoShan/LIO-SAM.git
cd ..
catkin_make
```

## Using Docker
Build image (based on ROS1 Kinetic):

```bash
docker build -t liosam-kinetic-xenial .
```

Once you have the image, start a container as follows:

```bash
docker run --init -it -d \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  liosam-kinetic-xenial \
  bash
```


## Prepare lidar data

The user needs to prepare the point cloud data in the correct format for cloud deskewing, which is mainly done in "imageProjection.cpp". The two requirements are:
  - **Provide point time stamp**. LIO-SAM uses IMU data to perform point cloud deskew. Thus, the relative point time in a scan needs to be known. The up-to-date Velodyne ROS driver should output this information directly. Here, we assume the point time channel is called "time." The definition of the point type is located at the top of the "imageProjection.cpp." "deskewPoint()" function utilizes this relative time to obtain the transformation of this point relative to the beginning of the scan. When the lidar rotates at 10Hz, the timestamp of a point should vary between 0 and 0.1 seconds. If you are using other lidar sensors, you may need to change the name of this time channel and make sure that it is the relative time in a scan.
  - **Provide point ring number**. LIO-SAM uses this information to organize the point correctly in a matrix. The ring number indicates which channel of the sensor that this point belongs to. The definition of the point type is located at the top of "imageProjection.cpp." The up-to-date Velodyne ROS driver should output this information directly. Again, if you are using other lidar sensors, you may need to rename this information. Note that only mechanical lidars are supported by the package currently.

## Record vehicle state data for GP model of vehicle-terrain intercation 

  - **Joystick required **. 
  ros joystick input is used to control the vehicle manually while recording the state messages. 
  To begin the recording process, first RUN 
  ```
    roslaunch gplogger gplogger.launch
  ```
To enable data logging  
<< publish topic  /gp_data_logging as "true" >>
To Save data   
<< publish topic  /gp_data_save as "true" >>

<p align='center'>
    <img src="./config/doc/imu-transform.png" alt="drawing" width="800"/>
</p>
<p align='center'>
    <img src="./config/doc/imu-debug.gif" alt="drawing" width="800"/>
</p>


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
2023 ICRA, under review 


## Acknowledgement
I would like to express my sincere thanks to following
```
- Our 3D Simulation environment and the Gazebo vehicle model is based on Autorally research platform  (Goldfain, Brian, et al. "Autorally: An open platform for aggressive autonomous driving." IEEE Control Systems Magazine 39.1 (2019): 26-55.)  
```

```
- Elevation and traversability mapping modules used in preprocessing step are based on these awesome work. 
      ( P. Fankhauser, M. Bloesch, C. Gehring, M. Hutter, and R. Siegwart,
        “Robot-centric elevation mapping with uncertainty estimates,” in Mobile
          Service Robotics. World Scientific, 2014, pp. 433–440.) 
          (P. Fankhauser, M. Bloesch, and M. Hutter, “Probabilistic terrain
          mapping for mobile robots with uncertain localization,” IEEE Robotics
          and Automation Letters, vol. 3, no. 4, pp. 3019–3026, 2018.) 
      (T. H. Y. Leung, D. Ignatyev, and A. Zolotas, “Hybrid terrain
traversability analysis in off-road environments,” in 2022 8th International
Conference on Automation, Robotics and Applications (ICARA).
IEEE, 2022, pp. 50–56.)
 ```
