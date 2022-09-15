/*
 * TraversabilityMap.hpp
 *
 *  Created on: Jun 09, 2015
 *      Author: Martin Wermelinger
 *	 Institute: ETH Zurich, Autonomous Systems Lab
 */

#pragma once

// Traversability
#include <traversability_msgs/FootprintPath.h>
#include <traversability_msgs/TraversabilityResult.h>

// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>

// ROS
#include <filters/filter_chain.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>
#include <image_geometry/pinhole_camera_model.h>

// STD
#include <string>
#include <vector>

// Boost
#include <boost/thread/recursive_mutex.hpp>
#include <mutex>
// OpenCv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cv_bridge/cv_bridge.h>

namespace traversability_estimation {

// Traversability value bounds.
constexpr double traversabilityMinValue = 0.0;
constexpr double traversabilityMaxValue = 1.0;

/*!
 * The terrain traversability estimation core. Updates the traversbility map and
 * evaluates the traversability of single footprints on this map.
 */
class TraversabilityMap {
 public:
  /*!
   * Constructor.
   */
  TraversabilityMap(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
  virtual ~TraversabilityMap();

  /*!
   * Computes the traversability based on the elevation map.
   * Traversability is set between 0.0 and 1.0, where a value of 0.0 means not
   * traversable and 1.0 means fully traversable.
   * @return true if successful.
   */
  bool computeTraversability();

  /*!
   * Checks the traversability of a footprint path and returns the traversability.
   * @param[in] path the footprint path that has to be checked.
   * @param[in] publishPolygons says if checked polygon and untraversable polygon should be computed and published.
   * @param[out] result the traversability result.
   * @return true if successful.
   */
  bool checkFootprintPath(const traversability_msgs::FootprintPath& path, traversability_msgs::TraversabilityResult& result,
                          const bool publishPolygons = false);

  /*!
   * Computes the traversability of a footprint at each map cell position twice:
   * first oriented in x-direction, and second oriented according to the yaw angle.
   * @param[in] footprintYaw orientation of the footprint.
   * @return true if successful.
   */
  bool traversabilityFootprint(double footprintYaw);

  /*!
   * Computes the traversability of a circular footprint at each map cell position.
   * @param[in] radius the radius of the circular footprint.
   * @param[in] offset the offset used for radius inflation.
   * @return true if successful.
   */
  bool traversabilityFootprint(const double& radius, const double& offset);

  /*!
   * The filter chain is reconfigured with the actual parameter on the parameter server.
   * @return true if successful.
   */
  bool updateFilter();

  /*!
   * Set the traversability map from layers of a grid_map_msgs::GridMap.
   * @param[in] msg grid map with the layers of a traversability map.
   * @return true if successful.
   */
  bool setTraversabilityMap(const grid_map_msgs::GridMap& msg);

  
/*!
   * Set the global traversability map.
   * @param[in] msg grid map with the layers of a traversability map.
   * @return true if successful.
   */
  bool set_global_map();

  /*!
   * Set the elevation map from layers of a grid_map_msgs::GridMap.
   * @param[in] msg grid map with a layer 'elevation'.
   * @return true if successful.
   */
  bool setElevationMap(const grid_map_msgs::GridMap& msg);

  /*!
   * Get the traversability map.
   * @return the requested traversability map.
   */
  grid_map::GridMap getTraversabilityMap();

  /*!
   * Get the Terrain Submap.
   * @param[in] grid_map::Gridmap Full traversability Gridmap
   * @return the requested traversability map.
   */
  // grid_map::GridMap downsamplingMap(const grid_map::GridMap& traversabilityMap);
    grid_map::GridMap downsamplingMap(const grid_map::GridMap& traversabilityMap,double map_size,double shift_dist);



   /*!
   * Assign Cost to each grid according to its terrain type from semantic mask
   * @param[in] grid_map::Gridmap Terrain (Sub) Gridmap
   * @return the cost assigned terrain traversability map
   */
  grid_map::GridMap assignTerrainCost(const grid_map::GridMap& MapIn);

  /*!
   * Extract Terrain map single grid position relative to camera frame
   * @param[in] grid_map::Position Single Grid position relative to map frame(odom)
   * @return grid_map::Position Single Grid position relative to camera frame 
   */
  grid_map::Position3 extractSingleGridPosition(const grid_map::Position3& position);

  /*!
   * Extract Terrain map all grids' position relative to camera frame
   * @param[in] grid_map::GridMap& Terrain Grid Map
   * @return std::vector<cv::Point3d> Position of all grid positions relative to camera frame 
   */
  std::vector<cv::Point3d> extractAllGridPosition(const grid_map::GridMap& MapOut);

  /*!
   * Project grid position from Robot Frame onto Image Plane
   * @param[in] std::vector<cv::Point3d> Vector containing grid positions relative to CameraFrame
   * @return std::vector<cv::Point2d> Vector containing projected points in pixel coordinates (u,v)
   */
  std::vector<cv::Point2d> projectAllGridPosition(const std::vector<cv::Point3d>& GridPosCameraFrame_vector);

  /*!
   * Set current pose for shifting the local Submap
   */
  void setPose(const geometry_msgs::PoseStamped cur_pose_);


  /*!
   * For debugging, draw the projected points on the image
   */
  void drawPoints(const std::vector<cv::Point2d> GridPosPixel_vector);

  /*!
   * Resets the cached traversability values.
   */
  void resetTraversabilityFootprintLayers();

  /*!
   * Publishes the latest traversability map.
   */
  void publishTraversabilityMap();

  void publishGlobalTraversabilityMap();

  /*!
   * Publishes the latest terrain map.
   */
  void publishTerrainMap();

  /*!
   * Checks if the traversability map is initialized.
   * @return true if the traversability map is initialized.
   */
  bool traversabilityMapInitialized();

  /*!
   * Gets the frame id of the traversability map.
   * @return frame id of the traversability map.
   */
  std::string getMapFrameId() const;

  /*!
   * Gets the default traversability value of unknown regions in the map.
   * @return default traversability value of unknown regions in the map
   */
  double getDefaultTraversabilityUnknownRegions() const;

  /*!
   * Sets the default traversability value of unknown regions in the map.
   * @param[in] defaultTraversability new default traversability value of unknown regions in the map
   */
  void setDefaultTraversabilityUnknownRegions(const double& defaultTraversability);

  /*!
   * Restores the default traversability value of unknown regions in the map, which was read during initialization .
   */
  void restoreDefaultTraversabilityUnknownRegionsReadAtInit();

  /*!
   * Checks if map has a valid traversability value at the specified cell.
   * @param x X coordinate of the cell to check.
   * @param y Y coordinate of the cell to check.
   * @return true if map has a valid traversability value at the specified cell.
   */
  bool mapHasValidTraversabilityAt(double x, double y) const;

  /*!
   * Create layers of traversabilty map.
   * @param useRawMap switch between raw and fused map.
   * @return true if layers are creates.
   */
  bool createLayers(bool useRawMap);

  /*
   * Set Robot Position relative to Odom Frame.
   * @param[in] geometry_msgs::PointStamped
  */
  void setRobotPose(geometry_msgs::PointStamped position);

  /*
   * Set Camera Info with the latest camera_info topic
   * @param[in] const sensor_msgs::CameraInfoConstPtr& info_msg
  */
  void getCameraModel_MSG(const sensor_msgs::CameraInfoConstPtr& info_msg);

  /*
   * Set Camera Info with the latest camera_info topic
  */
  image_geometry::PinholeCameraModel getCameraModel();

  /*
   * Get latest semantic mask msg
   * @param[in] const sensor_msgs::ImageConstPtr& image_msg
  */
  void getSemanticMask_MSG(const sensor_msgs::ImageConstPtr& image_msg);

  /*
   * Get latest semantic mask 
   * @param[in] const sensor_msgs::ImageConstPtr& image_msg
  */
  cv::Mat getSemanticMask();
  

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Gets the traversability value of the submap defined by the polygon. Is true if the whole polygon is traversable.
   * @param[in] polygon polygon that defines submap of the traversability map.
   * @param[in] computeUntraversablePolygon true if untraversable polygon within submap checked for traversability should be computed.
   * @param[out] traversability traversability value of submap defined by the polygon, the traversability is the mean of each cell within
   *             the polygon.
   * @param[out] untraversablePolygon untraversable polygon within area checked for traversability.
   * @return true if the whole polygon is traversable, false otherwise.
   */
  bool isTraversable(const grid_map::Polygon& polygon, const bool& computeUntraversablePolygon, double& traversability,
                     grid_map::Polygon& untraversablePolygon);

  /*!
   * Gets the traversability value of the submap defined by the polygon. Is true if the whole polygon is traversable.
   * @param[in] polygon polygon that defines submap of the traversability map.
   * @param[out] traversability traversability value of submap defined by the polygon, the traversability.
   * is the mean of each cell within the polygon.
   * @return true if the whole polygon is traversable, false otherwise.
   */
  bool isTraversable(const grid_map::Polygon& polygon, double& traversability);

  /*!
   * Gets the traversability value of a circular footprint.
   * @param[in] center the center position of the footprint.
   * @param[in] radiusMax the maximum radius of the footprint.
   * @param[in] computeUntraversablePolygon true if untraversable polygon within submap checked for traversability should be computed.
   * @param[out] traversability traversability value of the footprint.
   * @param[out] untraversablePolygon untraversable polygon within area checked for traversability.
   * @param[in] radiusMin if set (not zero), footprint inflation is applied and radiusMin is the minimum
   * valid radius of the footprint.
   * @return true if the circular footprint is traversable, false otherwise.
   */
  bool isTraversable(const grid_map::Position& center, const double& radiusMax, const bool& computeUntraversablePolygon,
                     double& traversability, grid_map::Polygon& untraversablePolygon, const double& radiusMin = 0);

  /*!
   * Gets the traversability value of a circular footprint.
   * @param[in] center the center position of the footprint.
   * @param[in] radiusMax the maximum radius of the footprint.
   * @param[out] traversability traversability value of the footprint.
   * @param[in] radiusMin if set (not zero), footprint inflation is applied and radiusMin is the minimum
   * valid radius of the footprint.
   * @return true if the circular footprint is traversable, false otherwise.
   */
  bool isTraversable(const grid_map::Position& center, const double& radiusMax, double& traversability, const double& radiusMin = 0);

  /*!
   * Checks if the overall inclination of the robot on a line between two
   * positions is feasible.
   * @param[in] start first position of the line.
   * @param[in] end last position of the line.
   * @return true if the whole line has a feasible inclination, false otherwise.
   */
  bool checkInclination(const grid_map::Position& start, const grid_map::Position& end);

  /*!
   * Checks if the map is traversable, only regarding steps, at the position defined
   * by the map index.
   * Small ditches and holes are not detected as steps.
   * @param[in] index index of the map to check.
   * @return true if no step is detected, false otherwise.
   */
  bool checkForStep(const grid_map::Index& indexStep);

  /*!
   * Checks if the map is traversable, only regarding slope, at the position defined
   * by the map index.
   * Small local slopes are not detected as slopes.
   * @param[in] index index of the map to check.
   * @return true if traversable regarding slope, false otherwise.
   */
  bool checkForSlope(const grid_map::Index& index);

  /*!
   * Checks if the map is traversable, only regarding roughness, at the position defined
   * by the map index.
   * Small local roughness is still detected as traversable terrain.
   * @param[in] index index of the map to check.
   * @return true if traversable regarding roughness, false otherwise.
   */
  bool checkForRoughness(const grid_map::Index& index);

  /*!
   * Publishes the footprint polygon.
   * @param[in] polygon footprint polygon checked for traversability.
   * @param[in] zPosition height of the polygon.
   */
  void publishFootprintPolygon(const grid_map::Polygon& polygon, double zPosition = 0.0);

  /*!
   * Publishes the untraversable polygon.
   * @param[in] untraversablePolygon polygon indicating untraversable parts..
   * @param[in] zPosition height of the polygon.
   */
  void publishUntraversablePolygon(const grid_map::Polygon& untraversablePolygon, double zPosition = 0.0);

  /*!
   * Bounds the passed traversability value to respect the allowed bounds.
   * @param traversabilityValue value to bound.
   * @return bounder value
   */
  double boundTraversabilityValue(const double& traversabilityValue) const;

  /*!
   * Checks if the map is traversable, according to defined filters.
   * @param[in] index index of the map to check.
   * @return true if traversable for defined filters.
   */
  bool isTraversableForFilters(const grid_map::Index& index);

  /*!
   * Checks the traversability of a circular footprint path and returns the traversability.
   * @param[in] path the footprint path that has to be checked.
   * @param[in] publishPolygons says if checked polygon and untraversable polygon should be computed and published.
   * @param[out] result the traversability result.
   * @return true if successful.
   */
  bool checkCircularFootprintPath(const traversability_msgs::FootprintPath& path, const bool publishPolygons,
                                  traversability_msgs::TraversabilityResult& result);

  /*!
   * Checks the traversability of a polygonal footprint path and returns the traversability.
   * @param[in] path the footprint path that has to be checked.
   * @param[in] publishPolygons says if checked polygon and untraversable polygon should be computed and published.
   * @param[out] result the traversability result.
   * @return true if successful.
   */
  bool checkPolygonalFootprintPath(const traversability_msgs::FootprintPath& path, const bool publishPolygons,
                                   traversability_msgs::TraversabilityResult& result);

  /*!
   * Computes mean height from poses.
   * @param[in] poses vector of poses to compute mean height.
   * @return mean height of poses.
   */
  template <typename Type>
  double computeMeanHeightFromPoses(const std::vector<Type>& poses) const {
    auto meanHeight = 0.0;
    if (poses.size() != 0) {
      for (int i = 0; i < poses.size(); i++) {
        meanHeight += poses.at(i).position.z;
      }
      meanHeight /= poses.size();
    }

    return meanHeight;
  }

  //! ROS node handle.
  ros::NodeHandle& nodeHandle_;

  //! Id of the frame of the elevation map.
  std::string mapFrameId_;

  //! Id of the frame of the robot.
  std::string robotFrameId_;

  //! Publisher of the traversability map.
  ros::Publisher traversabilityMapPublisher_;
  //  Publishe of the global traversability map 
  ros::Publisher globalMapPublisher_;

  //! Publisher of the terrain map.
  ros::Publisher terrainMapPublisher_;

  //! Footprint publisher.
  ros::Publisher footprintPublisher_;

  //! Untraversable polygon publisher
  ros::Publisher untraversablePolygonPublisher_;

  //! Vertices of the footprint polygon in base frame.
  std::vector<geometry_msgs::Point32> footprintPoints_;

  //! Robot parameter
  double maxGapWidth_;
  double circularFootprintOffset_;  // TODO: get this with FootprintPath msg.
  double criticalStepHeight_;
  double add_terrain_type_cost;

  //! global map info 
  double global_map_size_;

  //! Default value for traversability of unknown regions.
  double traversabilityDefault_;
  double traversabilityDefaultReadAtInit_;

  //! Verify footprint for roughness.
  bool checkForRoughness_;

  //! Verify overall robot inclination.
  bool checkRobotInclination_;

  //! Traversability map types.
  const std::string traversabilityType_;
  const std::string slopeType_;
  const std::string stepType_;
  const std::string roughnessType_;
  const std::string robotSlopeType_;

  //! Filter Chain
  filters::FilterChain<grid_map::GridMap> filter_chain_;

  //! Traversability map.
  grid_map::GridMap traversabilityMap_;
  grid_map::GridMap GlobalMap_;
  std::vector<std::string> traversabilityMapLayers_;
  bool traversabilityMapInitialized_;


  geometry_msgs::PoseStamped cur_pose;

  //! Elevation map.
  grid_map::GridMap elevationMap_;
  std::vector<std::string> elevationMapLayers_;
  bool elevationMapInitialized_;

  //! Terrain Submap
  grid_map::GridMap terrainMap_;
  std::vector<std::string> terrainMapLayers_;
  bool terrainMapInitialized_;

  //! Mutex lock for traversability map.
  mutable boost::recursive_mutex traversabilityMapMutex_;
  mutable boost::recursive_mutex elevationMapMutex_;
  mutable boost::recursive_mutex terrainMapMutex_;

  //! Z-position of the robot pose belonging to this map.
  double zPosition_;

  // Useful Grid Position in Odom Frame (map frame)
  std::vector<cv::Point3d> odom_frame_3d_position_;
  std::vector<grid_map::Position3> GridPosOdomFrame_vector_;
  std::vector<grid_map::Position3> filtered_GridPosOdomFrame_vector_;

  //! Center point of the elevation map.
  geometry_msgs::PointStamped robotPos_relative_to_odom_;

  //! TF listener.
  tf::TransformListener transformListener_;

  // Semantic mask
  cv::Mat semantic_mask_;
  mutable boost::recursive_mutex semanticMaskMutex_;

  std::mutex semantic_mask_msg_mutex_;
  sensor_msgs::ImageConstPtr last_semantic_mask_msg_;

  std::mutex camera_info_msg_mutex_;
  sensor_msgs::CameraInfoConstPtr last_camera_info_msg_;

  //! Camera Info
  image_geometry::PinholeCameraModel cam_model_;
  // Projection Matrix (3X4)
  //cv::Mat P(3,4,cv::DataType<double>::type);
  //cv::Mat P = cv::Mat::eye(3,4,cv::DataType<double>::type);
  cv::Mat P_ = (cv::Mat_<double>(3,4) << 2766.880127,0.0,970.500272,0.0,0.0,2790.281982,625.218685,0.0,0.0,0.0,1.0,0.0);

  // Camera Instrinsic
  cv::Mat K_ = (cv::Mat_<double>(3,3) << 2813.643275, 0.0, 969.285772, 0.0, 2808.326079, 624.049972, 0.0, 0.0, 1.0);
  // Rotational Matrix
  cv::Mat ROT_ = (cv::Mat_<double>(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  // Rodrigues Rotation <atrix
  cv::Mat Rodrigues_ROT_ = (cv::Mat_<double>(3,3));
  // Translation Vector (Homogeneous)
  cv::Mat Thomogeneous_= (cv::Mat_<double>(4,1));
  // Translation Vector (3X1)
  cv::Mat T_ = (cv::Mat_<double>(3,1) << 0.0,0.0,0.0); 

  // Distortion Coefficients  :: distortion_model: "plumb_bob"
  cv::Mat distCoeffs_ = (cv::Mat_<double>(5,1) << -0.134313,-0.025905,0.002181,0.00084,0.0);


  //! Semantic Mask Color Mapping [RGB]
  std::array<int,3> dirt {108,64,20};
  std::array<int,3> grass {0,102,0};
  std::array<int,3> tree {0,255,0};
  std::array<int,3> pole {0,153,153};
  std::array<int,3> water {0,128,255};
  std::array<int,3> sky {0,0,255};

  std::array<int,3> vehicle {255,255,0};
  std::array<int,3> object {255,0,127};
  std::array<int,3> asphalt {64,64,64};
  std::array<int,3> building {255,0,0};
  std::array<int,3> log {102,0,0};

  std::array<int,3> person {204,153,255};
  std::array<int,3> fence {102,0,204};
  std::array<int,3> bush {255,153,204};
  std::array<int,3> concrete {170,170,170};
  std::array<int,3> barrier {41,121,255};

  std::array<int,3> puddle {134,255,239};
  std::array<int,3> mud {99,66,34};
  std::array<int,3> rubble {110,22,138};
  std::array<int,3> ddirt {0,153,153};
  std::array<int,3> untraversable {255,255,255};


};

}  // namespace traversability_estimation
