#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>

#include "parameters.h"
#include "System.h"

typedef struct{
  double timestamp;
  cv::Mat image;
}IMG_TIME;

class ImageProcessor {
public:

  ImageProcessor();

  ImageProcessor(const ImageProcessor&) = delete;
  ImageProcessor operator=(const ImageProcessor&) = delete;

  ~ImageProcessor();
  void stereoCallback(const cv::Mat& cam0_img,const cv::Mat& cam1_img, std::vector<ImuConstPtr>& imu_buffer,double t);

  bool initialize(const std::string& sConfig_files);
  std::shared_ptr<IMG_MSG> featureUpdateCallback(double dStampSec);

  typedef boost::shared_ptr<ImageProcessor> Ptr;
  typedef boost::shared_ptr<const ImageProcessor> ConstPtr;

private:

  struct ProcessorConfig {
    int grid_row;
    int grid_col;
    int grid_min_feature_num;
    int grid_max_feature_num;

    int pyramid_levels;
    int patch_size;
    int fast_threshold;
    int max_iteration;
    double track_precision;
    double ransac_threshold;
    double stereo_threshold;
  };

  typedef unsigned long long int FeatureIDType;

  struct FeatureMetaData {
    FeatureIDType id;
    float response;
    int lifetime;
    cv::Point2f cam0_point;
    cv::Point2f cam1_point;
  };

 //<网格索引，网格中特征点集合>
  typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

  static bool keyPointCompareByResponse(const cv::KeyPoint& pt1,const cv::KeyPoint& pt2) {
    return pt1.response > pt2.response;
  }

  static bool featureCompareByResponse(const FeatureMetaData& f1,const FeatureMetaData& f2) {
    return f1.response > f2.response;
  }

  static bool featureCompareByLifetime(const FeatureMetaData& f1,const FeatureMetaData& f2) {
    return f1.lifetime > f2.lifetime;
  }

  bool loadParameters(const std::string& config_files);
  //!!!
  //bool createRosIO();


  //!!!将imu callback取出
  //void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  void initializeFirstFrame();

  void trackFeatures(std::vector<ImuConstPtr>& imu_buffer);

  void addNewFeatures();

  void pruneGridFeatures();
  //!!!
  //void publish();

  // //无用api
  // void drawFeaturesMono();

  // void drawFeaturesStereo();

  void createImagePyramids();

  void integrateImuData(cv::Matx33f& cam0_R_p_c,cv::Matx33f& cam1_R_p_c,std::vector<ImuConstPtr>& imu_buffer);

  void predictFeatureTracking(const std::vector<cv::Point2f>& input_pts,const cv::Matx33f& R_p_c,
                              const cv::Vec4d& intrinsics,std::vector<cv::Point2f>& compenstated_pts);

  void twoPointRansac(const std::vector<cv::Point2f>& pts1,const std::vector<cv::Point2f>& pts2,
      const cv::Matx33f& R_p_c,const cv::Vec4d& intrinsics,const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,const double& inlier_error,const double& success_probability,std::vector<int>& inlier_markers);
  void undistortPoints(const std::vector<cv::Point2f>& pts_in,const cv::Vec4d& intrinsics,const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,std::vector<cv::Point2f>& pts_out,const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
      const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));
  void rescalePoints(std::vector<cv::Point2f>& pts1,std::vector<cv::Point2f>& pts2,float& scaling_factor);
  std::vector<cv::Point2f> distortPoints(const std::vector<cv::Point2f>& pts_in,const cv::Vec4d& intrinsics,
      const std::string& distortion_model,const cv::Vec4d& distortion_coeffs);

  void stereoMatch(const std::vector<cv::Point2f>& cam0_points,std::vector<cv::Point2f>& cam1_points,std::vector<unsigned char>& inlier_markers);

  template <typename T>
  void removeUnmarkedElements(const std::vector<T>& raw_vec,const std::vector<unsigned char>& markers,std::vector<T>& refined_vec) {
    if (raw_vec.size() != markers.size()) {
      printf("The input size of raw_vec(%lu) and markers(%lu) does not match...",
          raw_vec.size(), markers.size());
    }
    for (int i = 0; i < markers.size(); ++i) {
      if (markers[i] == 0) continue;
      refined_vec.push_back(raw_vec[i]);
    }
    return;
  }

  bool is_first_img;
  FeatureIDType next_feature_id;
  ProcessorConfig processor_config;
  cv::Ptr<cv::Feature2D> detector_ptr;
  //!!! imu相关
  // std::vector<IMU_MSG> imu_msg_buffer;
  // std::vector<IMU_MSG> imu_buffer;

  // Camera calibration parameters
  std::string cam0_distortion_model;
  cv::Vec2i cam0_resolution;
  cv::Vec4d cam0_intrinsics;
  cv::Vec4d cam0_distortion_coeffs;

  std::string cam1_distortion_model;
  cv::Vec2i cam1_resolution;
  cv::Vec4d cam1_intrinsics;
  cv::Vec4d cam1_distortion_coeffs;

  // Take a vector from cam frame to the IMU frame.
  cv::Matx33d R_cam0_imu;
  cv::Vec3d t_cam0_imu;
  cv::Matx33d R_cam1_imu;
  cv::Vec3d t_cam1_imu;

  // !!!!Previous and current images
  // boost::shared_ptr<cv::Mat> cam0_prev_img_ptr;
  // boost::shared_ptr<cv::Mat> cam0_curr_img_ptr;
  // boost::shared_ptr<cv::Mat> cam1_curr_img_ptr;
  std::shared_ptr<IMG_TIME> cam0_prev_img_ptr;
  std::shared_ptr<IMG_TIME> cam0_curr_img_ptr;
  std::shared_ptr<IMG_TIME> cam1_curr_img_ptr;
  double pre_time,cur_time;

  // Pyramids for previous and current image
  std::vector<cv::Mat> prev_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam1_pyramid_;

  // Features in the previous and current image.
  boost::shared_ptr<GridFeatures> prev_features_ptr;
  boost::shared_ptr<GridFeatures> curr_features_ptr;

  // Number of features after each outlier removal step.
  int before_tracking;
  int after_tracking;
  int after_matching;
  int after_ransac;

  // // Subscribers and publishers.
  // message_filters::Subscriber<
  //   sensor_msgs::Image> cam0_img_sub;
  // message_filters::Subscriber<
  //   sensor_msgs::Image> cam1_img_sub;
  // message_filters::TimeSynchronizer<
  //   sensor_msgs::Image, sensor_msgs::Image> stereo_sub;
  // ros::Subscriber imu_sub;
  // ros::Publisher feature_pub;
  // ros::Publisher tracking_info_pub;
  // image_transport::Publisher debug_stereo_pub;

  // Debugging
  std::map<FeatureIDType, int> feature_lifetime;
  void updateFeatureLifetime();
  void featureLifetimeStatistics();
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

#endif
