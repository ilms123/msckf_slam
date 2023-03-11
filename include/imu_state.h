#ifndef IMU_STATE_H
#define IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define GRAVITY_ACCELERATION 9.81

struct IMUState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int StateIDType;

  StateIDType id;
  static StateIDType next_id;
  double time;

  Eigen::Vector4d orientation;   // qiw 世界w-imu系的变换

  Eigen::Vector3d position;   // twi:imu帧在世界坐标系的表示

  Eigen::Vector3d velocity; // w系下

  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d acc_bias;

  Eigen::Matrix3d R_imu_cam0;  // Rc0_i
  Eigen::Vector3d t_cam0_imu;

  Eigen::Vector4d orientation_null;
  Eigen::Vector3d position_null;
  Eigen::Vector3d velocity_null;

  static double gyro_noise;
  static double acc_noise;
  static double gyro_bias_noise;
  static double acc_bias_noise;

  static Eigen::Vector3d gravity; // w下

  // Transformation offset from the IMU frame to
  // the body frame. The transformation takes a
  // vector from the IMU frame to the body frame.
  // The z axis of the body frame should point upwards.
  // Normally, this transform should be identity.
  static Eigen::Isometry3d T_imu_body;

  IMUState(): id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()) {}

  IMUState(const StateIDType& new_id): id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    gyro_bias(Eigen::Vector3d::Zero()),
    acc_bias(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()),
    velocity_null(Eigen::Vector3d::Zero()) {}
};

typedef IMUState::StateIDType StateIDType;

#endif 
