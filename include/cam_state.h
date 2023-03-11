#ifndef CAM_STATE_H
#define CAM_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>

#include "imu_state.h"

struct CAMState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StateIDType id;
  double time;
  Eigen::Vector4d orientation;   //Rcw
  Eigen::Vector3d position;   //twc

  //修改测量的雅可比，使观测矩阵有合适的零空间？？？why???
  Eigen::Vector4d orientation_null;
  Eigen::Vector3d position_null;

  static Eigen::Isometry3d T_cam0_cam1;  // Tc1c0

  CAMState(): id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d(0, 0, 0)) {}

  CAMState(const StateIDType& new_id ): id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero()) {}
};

typedef std::map<StateIDType, CAMState, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const StateIDType, CAMState> > > CamStateServer;

#endif
