#include <iostream>
#include <algorithm>
#include <set>
#include <eigen3/Eigen/Dense>
#include <algorithm>

#include "image_processor.h"

using namespace std;
using namespace cv;
using namespace Eigen;

ImageProcessor::ImageProcessor() :
  is_first_img(true),
  prev_features_ptr(new GridFeatures()),
  curr_features_ptr(new GridFeatures()) {
  return;
}

ImageProcessor::~ImageProcessor() {
  // destroyAllWindows();
  return;
}

bool ImageProcessor::initialize(const std::string& sConfig_files) {
  if (!loadParameters(sConfig_files)) return false;
  printf("Finish loading parameters...");

  // Create feature detector.
  detector_ptr = FastFeatureDetector::create(processor_config.fast_threshold);   // 10

  return true;
}

bool ImageProcessor::loadParameters(const std::string& sConfig_files){
  cv::FileStorage fs(sConfig_files,cv::FileStorage::READ);
  if(!fs.isOpened()){
    printf("failed to load config files ...");
    return false;
  }
  // cam0
  fs["distortion_model"]>>cam0_distortion_model;

  int img_width=static_cast<int>(fs["image_width"]);
  int img_height=static_cast<int>(fs["image_height"]);
  cam0_resolution=cv::Vec2i(img_width,img_height);

  cv::FileNode n_intrin_cam0=fs["cam0_projection_parameters"];
  double cma0_fx=static_cast<double>(n_intrin_cam0["fx"]);
  double cma0_fy=static_cast<double>(n_intrin_cam0["fy"]);
  double cma0_cx=static_cast<double>(n_intrin_cam0["cx"]);
  double cma0_cy=static_cast<double>(n_intrin_cam0["cy"]);
  cam0_intrinsics=cv::Vec4d(cma0_fx,cma0_fy,cma0_cx,cma0_cy);

  cv::FileNode n_distort_cam0=fs["cam0_distortion_parameters"];
  double cma0_k1=static_cast<double>(n_distort_cam0["k1"]);
  double cma0_k2=static_cast<double>(n_distort_cam0["k2"]);
  double cma0_p1=static_cast<double>(n_distort_cam0["p1"]);
  double cma0_p2=static_cast<double>(n_distort_cam0["p2"]);
  cam0_distortion_coeffs=cv::Vec4d(cma0_k1,cma0_k2,cma0_p1,cma0_p2);
  //cam1
  fs["distortion_model"]>>cam1_distortion_model;
  cam1_resolution=cam0_resolution;
  cv::FileNode n_intrin_cam1=fs["cam1_projection_parameters"];
  double cma1_fx=static_cast<double>(n_intrin_cam1["fx"]);
  double cma1_fy=static_cast<double>(n_intrin_cam1["fy"]);
  double cma1_cx=static_cast<double>(n_intrin_cam1["cx"]);
  double cma1_cy=static_cast<double>(n_intrin_cam1["cy"]);
  cam1_intrinsics=cv::Vec4d(cma1_fx,cma1_fy,cma1_cx,cma1_cy);

  cv::FileNode n_distort_cam1=fs["cam1_distortion_parameters"];
  double cma1_k1=static_cast<double>(n_distort_cam1["k1"]);
  double cma1_k2=static_cast<double>(n_distort_cam1["k2"]);
  double cma1_p1=static_cast<double>(n_distort_cam1["p1"]);
  double cma1_p2=static_cast<double>(n_distort_cam1["p2"]);
  cam1_distortion_coeffs=cv::Vec4d(cma1_k1,cma1_k2,cma1_p1,cma1_p2);
  //TODO:外参写入

  return true;
}

void ImageProcessor::stereoCallback(const cv::Mat& cam0_img,const cv::Mat& cam1_img,std::vector<ImuConstPtr>& imu_buffer,double t) {
  // // !!!delete
  // cam0_curr_img_ptr = cv_bridge::toCvShare(cam0_img,
  //     sensor_msgs::image_encodings::MONO8);
  // cam1_curr_img_ptr = cv_bridge::toCvShare(cam1_img,
  //     sensor_msgs::image_encodings::MONO8);

  cam0_curr_img_ptr->image=cam0_img;
  cam0_curr_img_ptr->timestamp=t;
  cam1_curr_img_ptr->image=cam1_img;
  cam1_curr_img_ptr->timestamp=t;
  cur_time=t;
  //用到时间

  // Build the image pyramids once since they're used at multiple places
  createImagePyramids();

  // Detect features in the first frame.
  if (is_first_img) {

    pre_time=cur_time;
    initializeFirstFrame();
    is_first_img = false;
  } 
  else {
    // Track the feature in the previous image.
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    trackFeatures(imu_buffer);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time).count();
    addNewFeatures();
    pruneGridFeatures();
  }

  cam0_prev_img_ptr = cam0_curr_img_ptr;
  prev_features_ptr = curr_features_ptr;
  std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

  // // Initialize the current features to empty vectors.
  // curr_features_ptr.reset(new GridFeatures());
  // for (int code = 0; code <
  //     processor_config.grid_row*processor_config.grid_col; ++code) {
  //   (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
  // }

  return;
}

shared_ptr<IMG_MSG> ImageProcessor::featureUpdateCallback(double dStampSec){
  shared_ptr<IMG_MSG> feature_points(new IMG_MSG()); 
  std::vector<FeatureIDType> curr_ids(0);
  std::vector<cv::Point2f> curr_cam0_points(0);
  std::vector<cv::Point2f> curr_cam1_points(0);
  for (const auto& grid_features : (*curr_features_ptr)) {
      for (const auto& feature : grid_features.second) {
          curr_ids.push_back(feature.id);
          curr_cam0_points.push_back(feature.cam0_point);
          curr_cam1_points.push_back(feature.cam1_point);
      }
  }
  std::vector<cv::Point2f> curr_cam0_points_undistorted(0);
  std::vector<cv::Point2f> curr_cam1_points_undistorted(0);
  undistortPoints(curr_cam0_points, cam0_intrinsics, cam0_distortion_model,cam0_distortion_coeffs, curr_cam0_points_undistorted);
  undistortPoints(curr_cam1_points, cam1_intrinsics, cam1_distortion_model,cam1_distortion_coeffs, curr_cam1_points_undistorted);

  feature_points->features.resize(curr_ids.size());
  for(int i=0;i<curr_ids.size();i++){
      FeatureUV feature_imsg;
      feature_imsg.id=curr_ids[i];
      feature_imsg.uv=Vector2d(curr_cam0_points[i].x,curr_cam0_points[i].y);
      feature_imsg.uvRight=Vector2d(curr_cam1_points[i].x,curr_cam1_points[i].y);
      feature_imsg.uv_undist=Vector2d(curr_cam0_points_undistorted[i].x,curr_cam0_points_undistorted[i].y);
      feature_imsg.uvRight_undist=Vector2d(curr_cam1_points_undistorted[i].x,curr_cam1_points_undistorted[i].y);
      feature_points->features.push_back(feature_imsg);
  }
  feature_points->timestamp=dStampSec;
  // feature_buff.push_back(feature_points);

  // Initialize the current features to empty vectors.
  curr_features_ptr.reset(new GridFeatures());
  for (int code = 0; code <processor_config.grid_row*processor_config.grid_col; ++code) {
    (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
  }
  return feature_points;
}

// void ImageProcessor::imuCallback(
//     const sensor_msgs::ImuConstPtr& msg) {
//   // Wait for the first image to be set.
//   if (is_first_img) return;
//   imu_msg_buffer.push_back(*msg);
//   return;
// }

void ImageProcessor::createImagePyramids() {
  cv::Mat curr_cam0_img=cam0_curr_img_ptr->image;
  cv::Mat curr_cam1_img=cam1_curr_img_ptr->image;

  buildOpticalFlowPyramid(
      curr_cam0_img, curr_cam0_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),    // 15
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,    // 3
      BORDER_CONSTANT, false);

  buildOpticalFlowPyramid(
      curr_cam1_img, curr_cam1_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,
      BORDER_CONSTANT, false);
}
/**
  第一帧双目初始化
  1、提取左图Fast特征点
  2、进行双目匹配，获得右图特征点
  3、划分网格，将左图特征点划分到对应网格中，并将网格中特征点按照响应值排序
  4、对于每个网格，添加最少的优质点到curr_features_ptr，作为当前帧特征点
*/
void ImageProcessor::initializeFirstFrame() {
  // Size of each grid.
  const Mat& img = cam0_curr_img_ptr->image;
  static int grid_height = img.rows / processor_config.grid_row;   // 4
  static int grid_width = img.cols / processor_config.grid_col;    // 5

  // Detect new features on the frist image.
  //1.提取左目图像的特征点,提取出的特征点为未经过畸变矫正的真实像素点
  vector<KeyPoint> new_features(0);
  detector_ptr->detect(img, new_features);

  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  //2.立体匹配获得右目的匹配点以及匹配上的内点,并提取出左右相机对应的内点和其响应
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);   //存放左侧图像的响应
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }

  // Group the features into grids
  //3.将左右目中对应点分配到对应的网格中
  GridFeatures grid_new_features;
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  // Sort the new features in each grid based on its response.
  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),
        &ImageProcessor::featureCompareByResponse);

  // Collect new features within each grid with high response.
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

    for (int k = 0; k < processor_config.grid_min_feature_num &&
        k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;
    }
  }
  return;
}
/*
通过旋转  获得预测的坐标
input_pts，compensated_pts：表示像素坐标
P2（相机坐标）=R21*P1（相机坐标）
*/
void ImageProcessor::predictFeatureTracking(const vector<cv::Point2f>& input_pts,const cv::Matx33f& R_p_c,
                                            const cv::Vec4d& intrinsics,vector<cv::Point2f>& compensated_pts) {
  if (input_pts.size() == 0) {
    compensated_pts.clear();
    return;
  }
  compensated_pts.resize(input_pts.size());

  cv::Matx33f K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);

  cv::Matx33f H = K * R_p_c * K.inv();

  for (int i = 0; i < input_pts.size(); ++i) {
    cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
    cv::Vec3f p2 = H * p1;
    compensated_pts[i].x = p2[0] / p2[2];
    compensated_pts[i].y = p2[1] / p2[2];
  }

  return;
}

/*
特征点跟踪
1、用IMU前一帧与当前帧的角速度平均值，乘上时间dt，计算旋转矩阵R
2、进行初始估计：前一帧特征点，通过R，得到当前帧特征点（只考虑旋转）
3、LK光流跟踪（前一帧-左，当前帧-左），剔除外点
4、将当前帧双目匹配，进一步剔除outlier点
5、两点ransac计算（前一帧-左，当前帧-左） （前一帧-右，当前帧-右）剔除outlier点，并计算最优t
6、更新curr_features_ptr，保存最终特征信息
*/
void ImageProcessor::trackFeatures(std::vector<ImuConstPtr>& imu_buffer) {
  static int grid_height =ROW / processor_config.grid_row;
  static int grid_width =COL / processor_config.grid_col;
  // Compute a rough relative rotation which takes a vector
  // from the previous frame to the current frame.
  //前一帧到当前帧的变换 Rcp
  Matx33f cam0_R_p_c;
  Matx33f cam1_R_p_c;
  integrateImuData(cam0_R_p_c, cam1_R_p_c,imu_buffer);

  // Organize the features in the previous image.
  vector<FeatureIDType> prev_ids(0);
  vector<int> prev_lifetime(0);
  vector<Point2f> prev_cam0_points(0);
  vector<Point2f> prev_cam1_points(0);

  for (const auto& item : *prev_features_ptr) {  //map<int,std::vector<FeatureMetaData>> GridFeatures
    for (const auto& prev_feature : item.second) {  //每个网格中特征数目
      prev_ids.push_back(prev_feature.id);
      prev_lifetime.push_back(prev_feature.lifetime);
      prev_cam0_points.push_back(prev_feature.cam0_point);
      prev_cam1_points.push_back(prev_feature.cam1_point);
    }
  }
  before_tracking = prev_cam0_points.size();

  if (prev_ids.size() == 0) return;

  // Track features using LK optical flow method.
  vector<Point2f> curr_cam0_points(0);
  vector<unsigned char> track_inliers(0);
  predictFeatureTracking(prev_cam0_points,cam0_R_p_c, cam0_intrinsics, curr_cam0_points);

  calcOpticalFlowPyrLK(prev_cam0_pyramid_, curr_cam0_pyramid_,prev_cam0_points, curr_cam0_points,
      track_inliers, noArray(),Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
      processor_config.max_iteration,processor_config.track_precision),cv::OPTFLOW_USE_INITIAL_FLOW);

  for (int i = 0; i < curr_cam0_points.size(); ++i) {
    if (track_inliers[i] == 0) continue;
    if (curr_cam0_points[i].y < 0 ||curr_cam0_points[i].y > cam0_curr_img_ptr->image.rows-1 ||
        curr_cam0_points[i].x < 0 ||curr_cam0_points[i].x > cam0_curr_img_ptr->image.cols-1)
      track_inliers[i] = 0;
  }

  // Collect the tracked points.
  vector<FeatureIDType> prev_tracked_ids(0);
  vector<int> prev_tracked_lifetime(0);
  vector<Point2f> prev_tracked_cam0_points(0);
  vector<Point2f> prev_tracked_cam1_points(0);
  vector<Point2f> curr_tracked_cam0_points(0);

  removeUnmarkedElements(prev_ids, track_inliers, prev_tracked_ids);
  removeUnmarkedElements(prev_lifetime, track_inliers, prev_tracked_lifetime);
  removeUnmarkedElements(prev_cam0_points, track_inliers, prev_tracked_cam0_points);
  removeUnmarkedElements(prev_cam1_points, track_inliers, prev_tracked_cam1_points);
  removeUnmarkedElements(curr_cam0_points, track_inliers, curr_tracked_cam0_points);

  after_tracking = curr_tracked_cam0_points.size();


  // Outlier removal involves three steps, which forms a close
  // loop between the previous and current frames of cam0 (left)
  // and cam1 (right). Assuming the stereo matching between the
  // previous cam0 and cam1 images are correct, the three steps are:
  //
  // prev frames cam0 ----------> cam1
  //              |                |
  //              |ransac          |ransac
  //              |   stereo match |
  // curr frames cam0 ----------> cam1
  //
  // 1) Stereo matching between current images of cam0 and cam1.
  // 2) RANSAC between previous and current images of cam0.
  // 3) RANSAC between previous and current images of cam1.
  //
  // For Step 3, tracking between the images is no longer needed.
  // The stereo matching results are directly used in the RANSAC.

  // Step 1: stereo matching.
  vector<Point2f> curr_cam1_points(0);
  vector<unsigned char> match_inliers(0);
  stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

  vector<FeatureIDType> prev_matched_ids(0);
  vector<int> prev_matched_lifetime(0);
  vector<Point2f> prev_matched_cam0_points(0);
  vector<Point2f> prev_matched_cam1_points(0);
  vector<Point2f> curr_matched_cam0_points(0);
  vector<Point2f> curr_matched_cam1_points(0);

  removeUnmarkedElements(prev_tracked_ids, match_inliers, prev_matched_ids);
  removeUnmarkedElements(prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
  removeUnmarkedElements(prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
  removeUnmarkedElements(prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
  removeUnmarkedElements(curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
  removeUnmarkedElements(curr_cam1_points, match_inliers, curr_matched_cam1_points);

  after_matching = curr_matched_cam0_points.size();

  // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.获得内外点
  vector<int> cam0_ransac_inliers(0);
  twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, processor_config.ransac_threshold,0.99, cam0_ransac_inliers);

  vector<int> cam1_ransac_inliers(0);
  twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, processor_config.ransac_threshold,0.99, cam1_ransac_inliers);

  after_ransac = 0;

  for (int i = 0; i < cam0_ransac_inliers.size(); ++i) {
    if (cam0_ransac_inliers[i] == 0 ||cam1_ransac_inliers[i] == 0) continue;
    int row = static_cast<int>(curr_matched_cam0_points[i].y / grid_height);
    int col = static_cast<int>(curr_matched_cam0_points[i].x / grid_width);
    int code = row*processor_config.grid_col + col;
    (*curr_features_ptr)[code].push_back(FeatureMetaData());

    FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
    grid_new_feature.id = prev_matched_ids[i];
    grid_new_feature.lifetime = ++prev_matched_lifetime[i];
    grid_new_feature.cam0_point = curr_matched_cam0_points[i];
    grid_new_feature.cam1_point = curr_matched_cam1_points[i];

    ++after_ransac;
  }

  int prev_feature_num = 0;
  for (const auto& item : *prev_features_ptr)
    prev_feature_num += item.second.size();

  int curr_feature_num = 0;
  for (const auto& item : *curr_features_ptr)
    curr_feature_num += item.second.size();

  // printf(0.5,
  //     "\033[0;32m candidates: %d; track: %d; match: %d; ransac: %d/%d=%f\033[0m",
  //     before_tracking, after_tracking, after_matching,
  //     curr_feature_num, prev_feature_num,
  //     static_cast<double>(curr_feature_num)/
  //     (static_cast<double>(prev_feature_num)+1e-5));

  return;
}
/*
立体匹配：
1.利用两相机外参，获得右相机的特征点；
2.LK光流跟踪，计算outlier点；
3.剔除outier点：图像区域外的点剔除；计算E=t^R,计算极线方程，剔除误差过大的点（在无畸变  归一化平面上进行畸变矫正）
*/
void ImageProcessor::stereoMatch(const vector<cv::Point2f>& cam0_points,vector<cv::Point2f>& cam1_points,vector<unsigned char>& inlier_markers) {

  if (cam0_points.size() == 0) return;

  if(cam1_points.size() == 0) {

    const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;    //R10=（Rimu1）^T *Rimu0
    vector<cv::Point2f> cam0_points_undistorted;
    undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,cam0_distortion_coeffs, cam0_points_undistorted,R_cam0_cam1);
    cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics,cam1_distortion_model, cam1_distortion_coeffs);
  }

  // Track features using LK optical flow method.
  calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_,cam0_points, cam1_points,inlier_markers,
      noArray(),Size(processor_config.patch_size, processor_config.patch_size),processor_config.pyramid_levels,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,processor_config.max_iteration,processor_config.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  for (int i = 0; i < cam1_points.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    if (cam1_points[i].y < 0 ||cam1_points[i].y > cam1_curr_img_ptr->image.rows-1 ||
        cam1_points[i].x < 0 ||cam1_points[i].x > cam1_curr_img_ptr->image.cols-1)
      inlier_markers[i] = 0;
  }

  const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;  // R10
  const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu-t_cam1_imu);

  const cv::Matx33d t_cam0_cam1_hat(
      0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
      t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
      -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
  const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

  vector<cv::Point2f> cam0_points_undistorted(0);
  vector<cv::Point2f> cam1_points_undistorted(0);
  undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,cam0_distortion_coeffs, cam0_points_undistorted);
  undistortPoints(cam1_points, cam1_intrinsics, cam1_distortion_model,cam1_distortion_coeffs, cam1_points_undistorted);

  double norm_pixel_unit = 4.0 / (cam0_intrinsics[0]+cam0_intrinsics[1]+cam1_intrinsics[0]+cam1_intrinsics[1]);

  for (int i = 0; i < cam0_points_undistorted.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cv::Vec3d pt0(cam0_points_undistorted[i].x,cam0_points_undistorted[i].y, 1.0);
    cv::Vec3d pt1(cam1_points_undistorted[i].x,cam1_points_undistorted[i].y, 1.0);
    cv::Vec3d epipolar_line = E * pt0;
    double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(epipolar_line[0]*epipolar_line[0]+epipolar_line[1]*epipolar_line[1]);
    if (error > processor_config.stereo_threshold*norm_pixel_unit)   //5 
      inlier_markers[i] = 0;
  }

  return;
}

/*
添加新的特征点：
1.提取当前帧-左特征点：建立掩模，已存在特征点附近不再提取；
2.立体匹配 ，剔除外点；
3.将新提取的特征点存入curr_features_ptr，各个网格满足最少点
*/
void ImageProcessor::addNewFeatures() {

  const Mat& curr_img = cam0_curr_img_ptr->image;
  static int grid_height =cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));

  for (const auto& features : *curr_features_ptr) {
    for (const auto& feature : features.second) {
      const int y = static_cast<int>(feature.cam0_point.y);
      const int x = static_cast<int>(feature.cam0_point.x);

      int up_lim = y-2, bottom_lim = y+3, left_lim = x-2, right_lim = x+3;
      if (up_lim < 0) up_lim = 0;
      if (bottom_lim > curr_img.rows) bottom_lim = curr_img.rows;
      if (left_lim < 0) left_lim = 0;
      if (right_lim > curr_img.cols) right_lim = curr_img.cols;

      Range row_range(up_lim, bottom_lim);
      Range col_range(left_lim, right_lim);
      mask(row_range, col_range) = 0;
    }
  }

  vector<KeyPoint> new_features(0);
  detector_ptr->detect(curr_img, new_features, mask);

  // Collect the new detected features based on the grid.
  // Select the ones with top response within each grid afterwards.
  vector<vector<KeyPoint> > new_feature_sieve(processor_config.grid_row*processor_config.grid_col);
  for (const auto& feature : new_features) {
    int row = static_cast<int>(feature.pt.y / grid_height);
    int col = static_cast<int>(feature.pt.x / grid_width);
    new_feature_sieve[row*processor_config.grid_col+col].push_back(feature);
  }

  new_features.clear();
  for (auto& item : new_feature_sieve) {
    if (item.size() > processor_config.grid_max_feature_num) {
      std::sort(item.begin(), item.end(),&ImageProcessor::keyPointCompareByResponse);
      item.erase(item.begin()+processor_config.grid_max_feature_num, item.end());
    }
    new_features.insert(new_features.end(), item.begin(), item.end());
  }
  int detected_new_features = new_features.size();
  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }

  int matched_new_features = cam0_inliers.size();

  if (matched_new_features < 5 &&static_cast<double>(matched_new_features)/static_cast<double>(detected_new_features) < 0.1)
    printf("Images at [%f] seems unsynced...",cur_time);

  // Group the features into grids
  GridFeatures grid_new_features;
  for (int code = 0; code <processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),&ImageProcessor::featureCompareByResponse);

  int new_added_feature_num = 0;
  // Collect new features within each grid with high response.
  for (int code = 0; code <processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];
    if (features_this_grid.size() >=processor_config.grid_min_feature_num) // 3
      continue;    
    int vacancy_num = processor_config.grid_min_feature_num -features_this_grid.size();
    for (int k = 0;k < vacancy_num && k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;
      ++new_added_feature_num;
    }
  }
  return;
}

void ImageProcessor::pruneGridFeatures() {
  for (auto& item : *curr_features_ptr) {
    auto& grid_features = item.second;

    if (grid_features.size() <=processor_config.grid_max_feature_num)  //4
      continue;  
    std::sort(grid_features.begin(), grid_features.end(),&ImageProcessor::featureCompareByLifetime);
    grid_features.erase(grid_features.begin()+processor_config.grid_max_feature_num,grid_features.end());
  }
  return;
}

void ImageProcessor::undistortPoints(const vector<cv::Point2f>& pts_in,const cv::Vec4d& intrinsics,
    const string& distortion_model,const cv::Vec4d& distortion_coeffs,vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,const cv::Vec4d &new_intrinsics) {

  if (pts_in.size() == 0) return;
  const cv::Matx33d K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);
  // R10 cam左 到 cam 右的变换
  const cv::Matx33d K_new(
      new_intrinsics[0], 0.0, new_intrinsics[2],
      0.0, new_intrinsics[1], new_intrinsics[3],
      0.0, 0.0, 1.0);
  //pts_out输出的是变换到右cam系下无畸变归一化坐标（畸变uv—畸变归一化—去畸变—右cma坐标
  if (distortion_model == "radtan") {
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,rectification_matrix, K_new);
  } 
  else if (distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,rectification_matrix, K_new);
  } 
  else {
    printf("The model %s is unrecognized, use radtan instead...",distortion_model.c_str());
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,rectification_matrix, K_new);
  }
  return;
}

vector<cv::Point2f> ImageProcessor::distortPoints(const vector<cv::Point2f>& pts_in,const cv::Vec4d& intrinsics,
    const string& distortion_model,const cv::Vec4d& distortion_coeffs) {

  const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                      0.0, intrinsics[1], intrinsics[3],
                      0.0, 0.0, 1.0);
  vector<cv::Point2f> pts_out;
  if (distortion_model == "radtan") {
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,distortion_coeffs, pts_out);
  } 
  else if (distortion_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
  } 
  else {
    printf("The model %s is unrecognized, using radtan instead...",distortion_model.c_str());
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,distortion_coeffs, pts_out);
  }
  return pts_out;
}
/*
1.找到前一帧和当前帧图像对应的IMU时间戳，计算dt和角速度均值w；
2.利用IMU的均值w计算相机在dt时间内的均值：w_cam=R_cam_IMU*w；
3.利用相机均值计算在dt时间内R_pre_cur
*/
void ImageProcessor::integrateImuData(Matx33f& cam0_R_p_c, Matx33f& cam1_R_p_c,vector<ImuConstPtr>& imu_buffer) {
  // Find the start and the end limit within the imu msg buffer.
  auto begin_iter = imu_buffer.begin();
  while (begin_iter != imu_buffer.end()) {
    //!!!修正
    if ((*begin_iter)->timestamp-cam0_prev_img_ptr->timestamp < -0.01)
      ++begin_iter;
    else
      break;
  }
  auto end_iter = begin_iter;
  while (end_iter != imu_buffer.end()) {
    //!!!修正
    
    if ((*end_iter)->timestamp-cam0_curr_img_ptr->timestamp < 0.005)
      ++end_iter;
    else
      break;
  }

  // Compute the mean angular velocity in the IMU frame.
  cv::Vec3f mean_ang_vel(0.0, 0.0, 0.0);
  for (auto iter = begin_iter; iter < end_iter; ++iter)
    mean_ang_vel += cv::Vec3f((*iter)->angular_velocity.x(),(*iter)->angular_velocity.y(), (*iter)->angular_velocity.z());

  if (end_iter-begin_iter > 0)
    mean_ang_vel *= 1.0f / (end_iter-begin_iter);

  // Transform the mean angular velocity from the IMU
  // frame to the cam0 and cam1 frames.
  cv::Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;   //等价于
  cv::Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

  // Compute the relative rotation.
  //!!!修正
  double dtime = cam0_curr_img_ptr->timestamp-cam0_prev_img_ptr->timestamp;
  Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);  // Rpc
  Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
  cam0_R_p_c = cam0_R_p_c.t();
  cam1_R_p_c = cam1_R_p_c.t();  //输出Rcp

  // Delete the useless and used imu messages.
  imu_buffer.erase(imu_buffer.begin(), end_iter);
  return;
}

void ImageProcessor::rescalePoints(vector<Point2f>& pts1, vector<Point2f>& pts2,float& scaling_factor) {
  scaling_factor = 0.0f;
//到中心点距离的和
  for (int i = 0; i < pts1.size(); ++i) {
    scaling_factor += sqrt(pts1[i].dot(pts1[i]));
    scaling_factor += sqrt(pts2[i].dot(pts2[i]));
  }
  //将其缩放到距离中心点（根号2）的距离
  scaling_factor = (pts1.size()+pts2.size()) /scaling_factor * sqrt(2.0f);

  for (int i = 0; i < pts1.size(); ++i) {
    pts1[i] *= scaling_factor;
    pts2[i] *= scaling_factor;
  }
  return;
}
/*
利用前后两帧之间满足极线约束p2^T*t^（R*p1）=0，ransac求解平移t，统计内点最多的状态
1.利用IMUJ计算的R，假设纯旋转，计算误差:
   error>50个像素判断为outlier点，如果inlier点数不足3个，返回
   error<1个像素，认为没有平移，对极约束不成立，返回
2.将点坐标缩放到统一尺度下，归一化的根号2倍的尺度下
3.利用对极约束(Rp2 x p1)·t=0，取2个点计算t，计算该t下的inlier点数，统计inlier最多的即为最终结果
*/
void ImageProcessor::twoPointRansac(const vector<Point2f>& pts1, const vector<Point2f>& pts2,const cv::Matx33f& R_p_c, const cv::Vec4d& intrinsics,  //fx,fy,cx,cy
    const std::string& distortion_model,const cv::Vec4d& distortion_coeffs,const double& inlier_error,const double& success_probability,vector<int>& inlier_markers) {

  if (pts1.size() != pts2.size())
    printf("Sets of different size (%lu and %lu) are used...",pts1.size(), pts2.size());
 //归一化相机平面上，一个像素点的大小  1/f*u
  double norm_pixel_unit = 2.0 / (intrinsics[0]+intrinsics[1]);  //2/(fx+fy)
  int iter_num = static_cast<int>(ceil(log(1-success_probability) / log(1-0.7*0.7)));

  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 1);

  // Undistort all the points.
  vector<Point2f> pts1_undistorted(pts1.size());
  vector<Point2f> pts2_undistorted(pts2.size());
  undistortPoints(pts1, intrinsics, distortion_model,distortion_coeffs, pts1_undistorted);
  undistortPoints(pts2, intrinsics, distortion_model,distortion_coeffs, pts2_undistorted);

  // Compenstate the points in the previous image with
  // the relative rotation.
  //pts1_undistorted:无畸变帧1通过R后变换到帧2的坐标
  //pts2_undistorted:无畸变帧2坐标
  for (auto& pt : pts1_undistorted) {
    Vec3f pt_h(pt.x, pt.y, 1.0f);
    //Vec3f pt_hc = dR * pt_h;
    Vec3f pt_hc = R_p_c * pt_h;
    pt.x = pt_hc[0];
    pt.y = pt_hc[1];
  }
  //变换到同一尺度下计算差值，变换到归一化尺度乘以根号2的尺度下
  float scaling_factor = 0.0f;
  rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
  norm_pixel_unit *= scaling_factor;

  // Compute the difference between previous and current points,
  // which will be used frequently later.
  //pts_diff：立体匹配点-IMU计算的点的差
  //计算 前一帧点R变换到当前帧的坐标-当前帧的点坐标的差值
  vector<Point2d> pts_diff(pts1_undistorted.size());
  for (int i = 0; i < pts1_undistorted.size(); ++i)
    pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

  /*
  内点标记：
  1.前后帧对应点距离>50.0*norm_pixel_unit   ,inlier_maekers=0;
  2.  1后内点数<3，直接返回
  3.纯旋转时，不可ransac，当前后帧对应点距离>inlier_error*norm_pixel_unit,标为外点
  4.
  */
  // Mark the point pairs with large difference directly.
  // BTW, the mean distance of the rest of the point pairs
  // are computed.
  double mean_pt_distance = 0.0;
  int raw_inlier_cntr = 0;
  for (int i = 0; i < pts_diff.size(); ++i) {
    double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
    // 25 pixel distance is a pretty large tolerance for normal motion.
    // However, to be used with aggressive motion, this tolerance should
    // be increased significantly to match the usage.
    if (distance > 50.0*norm_pixel_unit) {
      inlier_markers[i] = 0;
    } 
    else {
      mean_pt_distance += distance;
      ++raw_inlier_cntr;
    }
  }
  mean_pt_distance /= raw_inlier_cntr;

  // If the current number of inliers is less than 3, just mark
  // all input as outliers. This case can happen with fast
  // rotation where very few features are tracked.
  //内点数<3，旋转过快 极线匹配不可  返回
  if (raw_inlier_cntr < 3) {
    for (auto& marker : inlier_markers) marker = 0;
      return;
  }

  // Before doing 2-point RANSAC, we have to check if the motion
  // is degenerated, meaning that there is no translation between
  // the frames, in which case, the model of the RANSAC does not
  // work. If so, the distance between the matched points will
  // be almost 0.
  //只有纯旋转，不可用RANSAC，同时匹配点间的距离近乎为0（小于一个像素）
  //if (mean_pt_distance < inlier_error*norm_pixel_unit) {
  if (mean_pt_distance < norm_pixel_unit) {
    for (int i = 0; i < pts_diff.size(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (sqrt(pts_diff[i].dot(pts_diff[i])) >inlier_error*norm_pixel_unit)
        inlier_markers[i] = 0;
    }
    return;
  }

  // In the case of general motion, the RANSAC model can be applied.
  // The three column corresponds to tx, ty, and tz respectively.
  //利用极限约束  存储p2^T*t^（R*p1）=0
  MatrixXd coeff_t(pts_diff.size(), 3);
  for (int i = 0; i < pts_diff.size(); ++i) {
    coeff_t(i, 0) = pts_diff[i].y;
    coeff_t(i, 1) = -pts_diff[i].x;
    coeff_t(i, 2) = pts1_undistorted[i].x*pts2_undistorted[i].y -pts1_undistorted[i].y*pts2_undistorted[i].x;
  }

  vector<int> raw_inlier_idx;
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] != 0)
      raw_inlier_idx.push_back(i);
  }

  vector<int> best_inlier_set;
  double best_error = 1e10;
  srand(1);
  //ransac迭代计算t，即两对匹配点可进行计算
  for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx) {
    int select_idx1 = rand() % (raw_inlier_idx.size()-1);
    int select_idx_diff = rand() % (raw_inlier_idx.size()-2)+1;
    int select_idx2 = select_idx1+
      select_idx_diff<raw_inlier_idx.size() ?select_idx1+select_idx_diff :select_idx1+select_idx_diff-raw_inlier_idx.size();

    int pair_idx1 = raw_inlier_idx[select_idx1];
    int pair_idx2 = raw_inlier_idx[select_idx2];

    // Construct the model;
    /*
    coeff_t=A=（a1,a2,a3)==(y1-y2,-(x1-x2),x1y2-x2y1)
    A矩阵=（tx,ty,tt)=(y1-y2,-(x1-x2),x1y2-x2y1)
    */
    Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
    Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
    Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
    vector<double> coeff_l1_norm(3);
    //L1范数各个元素的绝对值之和
    coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
    coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
    coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
    int base_indicator = min_element(coeff_l1_norm.begin(),
        coeff_l1_norm.end())-coeff_l1_norm.begin();

    Vector3d model(0.0, 0.0, 0.0);//即为t
    if (base_indicator == 0) {
      Matrix2d A;
      A << coeff_ty, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_tx);
      model(0) = 1.0;
      model(1) = solution(0);
      model(2) = solution(1);
    }
    else if (base_indicator ==1) {
      Matrix2d A;
      A << coeff_tx, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_ty);
      model(0) = solution(0);
      model(1) = 1.0;
      model(2) = solution(1);
    }
    else {
      Matrix2d A;
      A << coeff_tx, coeff_ty;
      Vector2d solution = A.inverse() * (-coeff_tz);
      model(0) = solution(0);
      model(1) = solution(1);
      model(2) = 1.0;
    }

    // Find all the inliers among point pairs.
    //计算误差  AX=0 ,相当与求出X直接乘以系数
    VectorXd error = coeff_t * model;
    //误差小于一定值的点，加入内点集中
    //注意单位一致：像素
    vector<int> inlier_set;
    for (int i = 0; i < error.rows(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (std::abs(error(i)) < inlier_error*norm_pixel_unit)
        inlier_set.push_back(i);
    }

    // If the number of inliers is small, the current
    // model is probably wrong.
    if (inlier_set.size() < 0.2*pts1_undistorted.size())
      continue;

    // Refit the model using all of the possible inliers.
    VectorXd coeff_tx_better(inlier_set.size());
    VectorXd coeff_ty_better(inlier_set.size());
    VectorXd coeff_tz_better(inlier_set.size());
    for (int i = 0; i < inlier_set.size(); ++i) {
      coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
      coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
      coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
    }

    Vector3d model_better(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_ty_better, coeff_tz_better;
      Vector2d solution =(A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
      model_better(0) = 1.0;
      model_better(1) = solution(0);
      model_better(2) = solution(1);
    }
    else if (base_indicator ==1) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_tz_better;
      Vector2d solution =(A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
      model_better(0) = solution(0);
      model_better(1) = 1.0;
      model_better(2) = solution(1);
    }
    else {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_ty_better;
      Vector2d solution =(A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
      model_better(0) = solution(0);
      model_better(1) = solution(1);
      model_better(2) = 1.0;
    }

    // Compute the error and upate the best model if possible.
    VectorXd new_error = coeff_t * model_better;

    double this_error = 0.0;
    for (const auto& inlier_idx : inlier_set)
      this_error += std::abs(new_error(inlier_idx));
    this_error /= inlier_set.size();

    if (inlier_set.size() > best_inlier_set.size()) {
      best_error = this_error;
      best_inlier_set = inlier_set;
    }
  }

  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 0);
  for (const auto& inlier_idx : best_inlier_set)
    inlier_markers[inlier_idx] = 1;

  return;
}


void ImageProcessor::updateFeatureLifetime() {
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features = (*curr_features_ptr)[code];
    for (const auto& feature : features) {
      if (feature_lifetime.find(feature.id) == feature_lifetime.end())
        feature_lifetime[feature.id] = 1;
      else
        ++feature_lifetime[feature.id];
    }
  }

  return;
}

void ImageProcessor::featureLifetimeStatistics() {

  map<int, int> lifetime_statistics;
  for (const auto& data : feature_lifetime) {
    if (lifetime_statistics.find(data.second) ==
        lifetime_statistics.end())
      lifetime_statistics[data.second] = 1;
    else
      ++lifetime_statistics[data.second];
  }

  for (const auto& data : lifetime_statistics)
    cout << data.first << " : " << data.second << endl;

  return;
}

