#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"


  /*
  Feature Salient part of an image. Please refer to the Appendix of "A Multi-State Constraint Kalman
  Filter for Vision-aided Inertial Navigation" for how the 3d position of a feature is initialized.
  */
struct Feature {

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int FeatureIDType;

  //OptimizationConfig Configuration parameters for 3d feature position optimization.
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,const Eigen::Vector3d& x, 
    const Eigen::Vector2d& z,double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @param cam_states : input camera poses.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion(const CamStateServer& cam_states) const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @param cam_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition(const CamStateServer& cam_states);

  FeatureIDType id;
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
    Eigen::aligned_allocator<std::pair<const StateIDType, Eigen::Vector4d> > > observations;

  Eigen::Vector3d position;

  bool is_initialized;

  static double observation_noise;

  static OptimizationConfig optimization_config;
};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<std::pair<const FeatureIDType, Feature> > > MapServer;


void Feature::cost(const Eigen::Isometry3d& T_c0_ci,const Eigen::Vector3d& x, 
    const Eigen::Vector2d& z,double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);  //逆深度
  //参考 https://zhuanlan.zhihu.com/p/77040286
  Eigen::Vector3d h = T_c0_ci.linear()*Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

/**
 * 重投影误差模型的Jacobian、残差，优化参数为首帧的特征点坐标，形式是(x/z,y/z,1/z)，稍微构造了一下
 * 已知c0的相机点（优化变量），ci帧的归一化点，相对位姿T_c0_ci，计算重投影误差，后面用Jacobian优化这个误差，得到最优的c0相机点坐标
 * https://zhuanlan.zhihu.com/p/77040286
*/
void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,const Eigen::Vector3d& x, 
  const Eigen::Vector2d& z,Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,double& w) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);  // 1/depth
  //计算第i帧对应的归一化点
  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)  // 0.01
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);

  return;
}
//已知归一化相机匹配点，相对位姿变换，三角化计算深度，得到相机坐标p
void Feature::generateInitialGuess(const Eigen::Isometry3d& T_c1_c2, 
  const Eigen::Vector2d& z1,const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.  使用正规方程求解
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}
/*
检查最大两帧的平移量，>阈值 为true
*/
bool Feature::checkMotion(const CamStateServer& cam_states) const {

  const StateIDType& first_cam_id = observations.begin()->first;
  const StateIDType& last_cam_id = (--observations.end())->first;
  //计算第一与最后相机的位姿（相对于世界坐标系）
  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = quaternionToRotation(cam_states.find(first_cam_id)->second.orientation).transpose();
  first_cam_pose.translation() =cam_states.find(first_cam_id)->second.position;

  Eigen::Isometry3d last_cam_pose;
  last_cam_pose.linear() = quaternionToRotation(cam_states.find(last_cam_id)->second.orientation).transpose();
  last_cam_pose.translation() =cam_states.find(last_cam_id)->second.position;

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  // 计算该特征点在首帧观测帧下的观测单位向量，转换到世界坐标系
  Eigen::Vector3d feature_direction(
      observations.begin()->second(0),
      observations.begin()->second(1), 1.0);      //相对于归一化平面的方向
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear()*feature_direction;   //相对于世界坐标系的方向

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  /*
  计算平移量在feature_direction上和垂直方向上的投影；
  相机沿z轴运动，feature_direction上投影很大，垂直方向上投影很小，左右移动时（垂直于z轴），相反
  */
  Eigen::Vector3d translation = last_cam_pose.translation() -first_cam_pose.translation();
  double parallel_translation =translation.transpose()*feature_direction;
  Eigen::Vector3d orthogonal_translation = translation -parallel_translation*feature_direction;

  if (orthogonal_translation.norm() >optimization_config.translation_threshold)   //0.4
    return true;
  else 
    return false;
}
/*
对feature的所有观测：
1.计算所有观测的坐标点与对应相机位姿；
2.将第一与最后一个观测三角化，计算三维点；
3.将三维点投影到所有相机，计算重投影误差；
4.LM优化

* 初始化特征点的世界坐标
 * 1、对当前特征点首帧观测帧、最后一帧观测帧，三角化计算特征点在首帧中的相机坐标
 * 2、构建重投影误差（所有观测帧与第一帧的重投影误差）模型，LM迭代优化该特征点的坐标，最后通过第一帧位姿转换到世界坐标系
*
*/
bool Feature::initializePosition(const CamStateServer& cam_states) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
  std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

  for (auto& m : observations) {
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input cam_states buffer.
    //观测点对应的camera_id
    auto cam_state_iter = cam_states.find(m.first);
    if (cam_state_iter == cam_states.end()) continue;

    // Add the measurement.
    measurements.push_back(m.second.head<2>());
    measurements.push_back(m.second.tail<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    //计算相机姿态相对于世界坐标系的变换 
    Eigen::Isometry3d cam0_pose;    //Twc0
    cam0_pose.linear() = quaternionToRotation(
        cam_state_iter->second.orientation).transpose();
    cam0_pose.translation() = cam_state_iter->second.position;

    Eigen::Isometry3d cam1_pose;   //Twc1
    cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();

    cam_poses.push_back(cam0_pose);
    cam_poses.push_back(cam1_pose);
  }

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  //将所有相对于世界坐标系的位姿变换到相对于观察到的第一个相机位姿
  Eigen::Isometry3d T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;   //Tcic0=Twci.inverse()*Twc0

  // Generate initial guess
  //第一个与最后一个观测点三角化计算坐标
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  //第一帧的左目观测与最后一帧的右目观测计算？
  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
      measurements[measurements.size()-1], initial_position);

  //solution(measurements[0]的u, v, 1/depth)
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));   // 1/d，为深度的逆

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;   //1e-3
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.
  //计算在每帧上的重投影误差
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {   // Tcic0
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.LM算法
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {   // Rcic0
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;
      //// cam_poses为Tcic0
      jacobian(cam_poses[i], solution, measurements[i], J, r, w);

      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);  // 10

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&  // 10
      delta_norm > optimization_config.estimation_precision);  // 5e-7

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d position =
      pose.linear()*final_position + pose.translation();
    if (position(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }

  // Convert the feature position to the world frame.
  position = T_c0_w.linear()*final_position + T_c0_w.translation();

  if (is_valid_solution)
    is_initialized = true;

  return is_valid_solution;
}

#endif 
