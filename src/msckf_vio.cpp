#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>

#ifdef USING_SPARSE_QR
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#endif


#include <boost/math/distributions/chi_squared.hpp>

#include "msckf_vio.h"
#include "math_utils.hpp"
#include "utils.h"

#include <thread>

using namespace std;
using namespace Eigen;

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table; 

MsckfVio::MsckfVio():
  is_gravity_set(false),
  is_first_img(true) {
  return;
}

bool MsckfVio::loadParameters(const std::string& sConfig_files) {
  // Frame id
  cv::FileStorage fs(sConfig_files,cv::FileStorage::READ);
  if(!fs.isOpened()){
    printf("failed to load config files ...");
    return false;
  }

  fs["fixed_frame_id"]>>fixed_frame_id;
  fs["child_frame_id"] >> child_frame_id;
  fs["frame_rate"]>> frame_rate;
  fs["position_std_threshold"]>>position_std_threshold;   //重置系统的位置阈值

  fs["rotation_threshold"]>>rotation_threshold;
  fs["translation_threshold"]>>translation_threshold ;
  fs["tracking_rate_threshold"]>> tracking_rate_threshold;

  // Feature optimization parameters
  fs["feature_translation_threshold"]>>Feature::optimization_config.translation_threshold;

  // Noise related parameters
  fs["noise_gyro"]>>IMUState::gyro_noise;
  fs["noise_acc"]>>IMUState::acc_noise;
  fs["noise_gyro_bias"]>>IMUState::gyro_bias_noise;
  fs["noise_acc_bias"]>> IMUState::acc_bias_noise;
  fs["noise_feature"]>>Feature::observation_noise;

  // Use variance instead of standard deviation. （即sigma^2）
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  // nh.param<double>("initial_state/velocity/x",
  //     state_server.imu_state.velocity(0), 0.0);
  // nh.param<double>("initial_state/velocity/y",
  //     state_server.imu_state.velocity(1), 0.0);
  // nh.param<double>("initial_state/velocity/z",
  //     state_server.imu_state.velocity(2), 0.0);


  // fs["initial_state_velocity"]>>state_server.imu_state.velocity
  state_server.imu_state.velocity=Eigen::Vector3d::Zero();
   //(q,bg,v,ba,p,thea,p)
  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  fs["max_cam_state_size"]>>max_cam_state_size;

  //!!!TODO外参写入
  // // Transformation offsets between the frames involved. 
  // Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu"); 
  // Isometry3d T_cam0_imu = T_imu_cam0.inverse();

  // state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  // state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  
  // CAMState::T_cam0_cam1 =
  //   utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
  // IMUState::T_imu_body =
  //   utils::getTransformEigen(nh, "T_imu_body").inverse();

  fs["max_cam_state_size"]>>max_cam_state_size;

  return true;
}
// //-----------------------------------------------------------------
// //输出中间轨迹内容
// #if 1
// static const int p_max_cnt = 10000;
// static const int p_cols = 8;
// static double *p_log_data = new double[p_max_cnt * p_cols];
// static int p_cnt = 0;
// void mylog()
// {
//     FILE* fp = fopen("/home/kiki/MyFiles/euroc_data/eva_data/ms_evo.txt","w");
//     int k = 0;
//     while(1)
//     {
//         if(k < p_cnt)
//         {
//             for(int i=0; i<p_cols; i++)
//             {
//                 if(i > 0) fprintf(fp, " ");
//                 fprintf(fp, "%f", p_log_data[p_cols * k + i]);
//             }
//             fprintf(fp, "\n");
//             fflush(fp);
//             k++;
//         }
//         else
//         {
//             usleep(100000);
//         }
//     }
//     fclose(fp);
// }
// static std::thread th_log(mylog);
// #endif
// //-----------------------------------------------------------------

bool MsckfVio::HasInitGravityAndBias(){
  return is_gravity_set;
}
/*
初始化：
1.完成参数加载；
2.初始化误差变换矩阵G；
3.初始化卡方分布的置信度；
4.创建ROS的IO
*/

bool MsckfVio::initialize(const std::string& sConfig_files) {
  if (!loadParameters(sConfig_files)) return false;
  printf("Finish loading ROS parameters...");
//continuous_noise_cov相当于x=F*X+G*n  中的N,噪声项
  // Initialize state server
  state_server.continuous_noise_cov =Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =Matrix3d::Identity()*IMUState::acc_bias_noise;

 /*卡方分布：1）衡量观测分布和理论分布的拟合程度；2）测量定性数据两个分类标准之间的独立性
 */
  // Initialize the chi squared test table with confidence
  // level 0.95.置信度水平0.95的卡方分布
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);   
    chi_squared_test_table[i] =boost::math::quantile(chi_squared_dist, 0.05);
  }
  return true;
}

/*
初始化时计算gyro_bias和IMU旋转四元数（世界系-IMU系的变换）
1.计算时间t内的角速度和和加速度和
2.gyro_bias=角速度和/时间  ，加速度=加速度和/时间  ，重力=（0,0,-加速度）；
3.根据 加速度与重力 计算IMU的旋转四元数（世界系-IMU系的变换）
*/
void MsckfVio::initializeGravityAndBias(std::vector<ImuConstPtr>& imu_buff) {

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();

  for (const auto& imu_msg : imu_buff) {

    Vector3d angular_vel = imu_msg->angular_velocity;
    Vector3d linear_acc = imu_msg->linear_acceleration;
    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }

  state_server.imu_state.gyro_bias =sum_angular_vel / imu_buff.size();
  Vector3d gravity_imu =sum_linear_acc / imu_buff.size();

  // Initialize the initial orientation, so that the estimation
  // is consistent with the inertial frame.
  double gravity_norm = gravity_imu.norm();
  IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);

  Quaterniond q0_i_w = Quaterniond::FromTwoVectors(gravity_imu, -IMUState::gravity);  //qwi
  // qiw 将imu坐标系对齐到世界坐标系
  state_server.imu_state.orientation =rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());
  return;
}

//！！！需要增加整个系统的reset
// /*
// 1.重置IMU的状态；
// 2.重置状态协方差和F；
// 3.清除特征和IMU缓存
// */
// bool MsckfVio::resetCallback(
//     std_srvs::Trigger::Request& req,
//     std_srvs::Trigger::Response& res) {

//   ROS_WARN("Start resetting msckf vio...");
//   // Temporarily shutdown the subscribers to prevent the
//   // state from updating.
//   feature_sub.shutdown();
//   imu_sub.shutdown();

//   // Reset the IMU state.
//   IMUState& imu_state = state_server.imu_state;
//   imu_state.time = 0.0;
//   imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
//   imu_state.position = Vector3d::Zero();
//   imu_state.velocity = Vector3d::Zero();
//   imu_state.gyro_bias = Vector3d::Zero();
//   imu_state.acc_bias = Vector3d::Zero();
//   imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
//   imu_state.position_null = Vector3d::Zero();
//   imu_state.velocity_null = Vector3d::Zero();

//   // Remove all existing camera states.
//   state_server.cam_states.clear();

//   // Reset the state covariance.
//   double gyro_bias_cov, acc_bias_cov, velocity_cov;
//   nh.param<double>("initial_covariance/velocity",
//       velocity_cov, 0.25);
//   nh.param<double>("initial_covariance/gyro_bias",
//       gyro_bias_cov, 1e-4);
//   nh.param<double>("initial_covariance/acc_bias",
//       acc_bias_cov, 1e-2);

//   double extrinsic_rotation_cov, extrinsic_translation_cov;
//   nh.param<double>("initial_covariance/extrinsic_rotation_cov",
//       extrinsic_rotation_cov, 3.0462e-4);
//   nh.param<double>("initial_covariance/extrinsic_translation_cov",
//       extrinsic_translation_cov, 1e-4);

//   state_server.state_cov = MatrixXd::Zero(21, 21);
//   for (int i = 3; i < 6; ++i)
//     state_server.state_cov(i, i) = gyro_bias_cov;
//   for (int i = 6; i < 9; ++i)
//     state_server.state_cov(i, i) = velocity_cov;
//   for (int i = 9; i < 12; ++i)
//     state_server.state_cov(i, i) = acc_bias_cov;
//   for (int i = 15; i < 18; ++i)
//     state_server.state_cov(i, i) = extrinsic_rotation_cov;
//   for (int i = 18; i < 21; ++i)
//     state_server.state_cov(i, i) = extrinsic_translation_cov;

//   // Clear all exsiting features in the map.
//   map_server.clear();

//   // Clear the IMU msg buffer.
//   imu_msg_buffer.clear();

//   // Reset the starting flags.
//   is_gravity_set = false;
//   is_first_img = true;

//   // Restart the subscribers.
//   imu_sub = nh.subscribe("imu", 100,
//       &MsckfVio::imuCallback, this);
//   feature_sub = nh.subscribe("features", 40,
//       &MsckfVio::featureCallback, this);

//   // TODO: When can the reset fail?
//   res.success = true;
//   ROS_WARN("Resetting msckf vio completed...");
//   return true;
// }

void MsckfVio::featureCallback(const ImgConstPtr img_msg_ptr,const std::vector<ImuConstPtr>& imu_buff) {

  if (!is_gravity_set) return;

  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = img_msg_ptr->timestamp;
  }

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  std::chrono::steady_clock::time_point processing_start_time =std::chrono::steady_clock::now();

  // Propogate the IMU state.
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  batchImuProcessing(img_msg_ptr->timestamp,imu_buff);
  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
  double imu_processing_time =std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time).count();

  // Augment the state vector.
  start_time = std::chrono::steady_clock::now();
  stateAugmentation(img_msg_ptr->timestamp);
  end_time = std::chrono::steady_clock::now();
  double state_augmentation_time =std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time).count();

  // Add new observations for existing features or new
  // features in the map server.
  start_time =std::chrono::steady_clock::now();
  addFeatureObservations(img_msg_ptr);
  end_time = std::chrono::steady_clock::now();
  double add_observations_time =std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time).count();

  // Perform measurement update if necessary.
  start_time = std::chrono::steady_clock::now();
  removeLostFeatures();
  end_time = std::chrono::steady_clock::now();
  double remove_lost_features_time =std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time).count();
  start_time =std::chrono::steady_clock::now();
  pruneCamStateBuffer();
  end_time = std::chrono::steady_clock::now();
  double prune_cam_states_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time).count();

  // Reset the system if necessary.
  onlineReset();

  std::chrono::steady_clock::time_point processing_end_time =std::chrono::steady_clock::now();
  double processing_time =std::chrono::duration_cast<std::chrono::duration<double>>(processing_end_time - processing_start_time).count();
  if (processing_time > 1.0/frame_rate) {
    ++critical_time_cntr;
    printf("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
    //printf("IMU processing time: %f/%f\n",
    //    imu_processing_time, imu_processing_time/processing_time);
    //printf("State augmentation time: %f/%f\n",
    //    state_augmentation_time, state_augmentation_time/processing_time);
    //printf("Add observations time: %f/%f\n",
    //    add_observations_time, add_observations_time/processing_time);
    printf("Remove lost features time: %f/%f\n",
        remove_lost_features_time, remove_lost_features_time/processing_time);
    printf("Remove camera states time: %f/%f\n",
        prune_cam_states_time, prune_cam_states_time/processing_time);
    //printf("Publish time: %f/%f\n",
    //    publish_time, publish_time/processing_time);
  }
//   //-------------------------------------------------------------------------
//   #if 1
//     const auto& s = state_server.imu_state;
//     const auto& p = s.position;
//     const auto& q = s.orientation;
//     const auto& v = s.velocity;
//     const auto& ba = s.acc_bias;
//     const auto& bg = s.gyro_bias;
//     double *lp = p_log_data + p_cnt * p_cols;
//     lp[0] = s.time;
//     lp[1] = p(0);
//     lp[2] = p(1);
//     lp[3] = p(2);
//     lp[4] = q.x();
//     lp[5] = q.y();
//     lp[6] = q.z();
//     lp[7] = q.w();
//     //lp[8] = v(0);
//     //lp[9] = v(1);
//     //lp[10] = v(2);
//     //lp[11] = ba(0);
//     //lp[12] = ba(1);
//     //lp[13] = ba(2);
//     //lp[14] = bg(0);
//     //lp[15] = bg(1);
//     //lp[16] = bg(2);
//     //lp[17] = processing_time;
//     p_cnt++;
//   #endif
// //--------------------------------------------------------------------------

  return;
}

/*
time_bound 当前图像帧时间戳
IMU状态传播：
1.更新p v q
2.更新协方差矩阵
*/
void MsckfVio::batchImuProcessing(const double& time_bound,const std::vector<ImuConstPtr>& imu_buff) {
  int used_imu_msg_cntr = 0;
  for (const auto& imu_msg : imu_buff) {
    double imu_time = imu_msg->timestamp;
    if (imu_time < state_server.imu_state.time) {  //buffer中imu某一帧的时间<imu记录的时间
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;
    Vector3d m_gyro=imu_msg->angular_velocity;
    Vector3d m_acc=imu_msg->linear_acceleration;
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }
  state_server.imu_state.id = IMUState::next_id++;
  // // Remove all used IMU msgs.
  // imu_msg_buffer.erase(imu_msg_buffer.begin(),
  //     imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}
/*
IMU状态传播：对state_server.imu_state.time---图像帧之间的每帧IMU进行处理
1.更新p v q  (q-四元数积分公式，v p-四阶Runge-Kutta公式)
2.计算连续变换矩阵F、G以及离散变换矩阵Phi；
3.OC-KF进行离散变换矩阵Phi修正，计算离散噪声协方差矩阵Q
4.更新系统状态协方差矩阵
*/
void MsckfVio::processModel(const double& time,const Vector3d& m_gyro,const Vector3d& m_acc) {

  IMUState& imu_state = state_server.imu_state;
  // 当前时刻角速度、加速度，去偏置  (w^=wm-bg^注：w^表示估计值)
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  //imu两帧之间的时间间隔
  //解释：time表示IMU buffer中记录的时间大于当前IMU状态时间的第一帧
  double dtime = time - imu_state.time;

  // Compute discrete transition and noise covariance matrix
  Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
  Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order,which can be considered to be accurate enough assuming dtime is within 0.01s.
  //误差离散状态模型  dx=F*X+G*n
  //离散时间状态转移矩阵 Phi = exp(Fdt) = I + Fdt + 1/2*Fdt*Fdt + 1/6*Fdt*Fdt*Fdt
  Matrix<double, 21, 21> Fdt = F * dtime;
  Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
  Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  predictNewState(dtime, gyro, acc);
//???？？修正离散变换矩阵的原理？？？？？？？
  // Modify the transition matrix
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  //类似于求解 A*x=E(单位向量)
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(dtime*imu_state.velocity_null+imu_state.position_null-imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  //Q连续时间噪声协方差矩阵
  Matrix<double, 21, 21> Q = Phi*G*state_server.continuous_noise_cov*G.transpose()*Phi.transpose()*dtime;
  state_server.state_cov.block<21, 21>(0, 0) =Phi*state_server.state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;
  //state_server.state_cov为整个系统的协方差矩阵
  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols()-21) =
      Phi * state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols()-21);
    state_server.state_cov.block(21, 0, state_server.state_cov.rows()-21, 21) =
      state_server.state_cov.block(1, 0, state_server.state_cov.rows()-21, 21) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}
/*计算v、p等状态向量时，不同的积分方法精度不同
1.相应的更新会放入state_server.imu_state中，影响后续计算
*/
void MsckfVio::predictNewState(const double& dt,const Vector3d& gyro,const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  //四元数微分
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  //四元数四元数的毕卡法更新
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  //当角增量很小时的近似，实部项没有做近似，虚部项使用了洛必达法则
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();
//4阶龙格库塔（Runge-Kutta）积分求速度和位置,假设前后两时刻加速度恒定
  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +IMUState::gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}
/*状态扩增
1.根据Rci（IMU-相机位姿）和Riw（世界位姿-IMU）计算当前相机位姿，加入到当前相机的状态向量
  state_server.cam_states中；
2.构建系统协方差增广矩阵J；
3.系统协方差矩阵扩充、更新
*/

void MsckfVio::stateAugmentation(const double& time) {

//R_i_c表示从i系到c系的变换矩阵 Rci
//t_c_i理解为i系下c坐标系的坐标 tic（与orbslam中位姿R-t中的t不同）
  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;
  //相机姿态直接由IMU最近帧确定，无校正过程
  // Add a new camera state to the state server.
  Matrix3d R_w_i = quaternionToRotation(state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i; //Rcw=Rci*Riw
  Vector3d t_c_w = state_server.imu_state.position + R_w_i.transpose()*t_c_i; // twc=twi+Rwi*tic

  state_server.cam_states[state_server.imu_state.id] =CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c); // Rcw
  cam_state.position = t_c_w;  // twc

  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = R_w_i.transpose();

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 21, 21>& P11 =state_server.state_cov.block<21, 21>(0, 0);
  const MatrixXd& P12 =state_server.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance. 
  //J*Pk
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  //Pk*J^T
  state_server.state_cov.block(0, old_cols, old_rows, 6) =state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
    //J*Pk*J^T   ???Pk=(Pl1,Pl2)  
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;
  return;
}

/*map_server为一个map，<feature.id,feature>
observations为一个map <state_server.imu_state.id,点坐标>
添加新观测的特征点：
1.对于map_server 已存在特征点：添加观测；
2.对于map_server 中未存在特征点：添加新的特征点
*/
void MsckfVio::addFeatureObservations(const ImgConstPtr img_msg_ptr) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : img_msg_ptr->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.uv_undist.x(), feature.uv_undist.y(),feature.uvRight_undist.x(), feature.uvRight_undist.y());
    } 
    else {
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.uv_undist.x(), feature.uv_undist.y(),feature.uvRight_undist.x(), feature.uvRight_undist.y());
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}
/*
计算单特征单相机的雅可比
？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
疑问：1.修正雅可比矩阵保证观测约束
2.残差r的计算有体现出计算J的必要性吗：有，更新了矩阵H_x H_f
？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
*/
void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {

  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);   // Rcw
  const Vector3d& t_c0_w = cam_state.position;  // twc

  // Cam1 pose.
  //?？通过两相机之间外参计算第二个相机R，t的原理
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0; // Rc1w
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  //？？？？？？？？？？？？修正雅可比矩阵保证观测约束？？？？？？？？？？？？？？？？
  Matrix<double, 4, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}
/*
feature被多个相机观测到的观测模型：
相当于计算：
rj=H_xj*Xc+H_fj*pj+n
H_xj：大小为4M*(21+6N)；Xc：状态误差向量 (21+6N)*1
H_fj：大小为4M*3；      pj：特征点xyz
投影到左零空间
r=H_x*Xc+n0
H_x：大小 （4M-3）*（21+6N）
*/
void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  //??是否多余，cam_state_ids为可观测到此特征的所有相机ID
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 4 * valid_cam_state_ids.size();
  // 4M,21+6N
  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      21+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    //feature被单个相机观测的模型
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);


    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  //A的大小为 4M*（4M-3）
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);  //Thin表示只要其一值个数的列
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);
//投影到H_fj的左零空间后，H_x的大小 （4M-3）*（21+6N）
  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
#ifdef USING_SPARSE_QR
    // Convert H to a sparse matrix.
    Eigen::SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    Eigen::SPQR<Eigen::SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    Eigen::MatrixXd H_temp;
    Eigen::VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);
#else
    HouseholderQR<MatrixXd> qr_helper(H);
    MatrixXd Q = qr_helper.householderQ();
    MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    H_thin = Q1.transpose() * H;
    r_thin = Q1.transpose() * r;
#endif
  } 
  else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  //为K计算公式中的T_h*P*T_h^T+Rn
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(   // 0.01
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<21>();
//判断imu的变化量，v p是否变化过大
  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    //return;
  }
  //更新imu中各个状态量
  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

bool MsckfVio::gatingTest(const MatrixXd& H, const VectorXd& r, const int& dof) {
//P1：（4M-3）*（4M-3）   类似于卡尔曼增益
  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  if (gamma < chi_squared_test_table[dof]) {
    return true;
  } 
  else {
    return false;
  }
}
 /*
 关键处理：
 1.找出无效的特征点，从滑窗中移除无效点；
 2.对当前帧跟丢的特征点，构建残差模型，进行EKF观测更新，从滑窗中删除跟丢点(限制雅可比矩阵的行数1500)；
 3.更新IMU状态、相机状态和系统协方差矩阵
  */
void MsckfVio::removeLostFeatures() {
  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);    //无效特征
  vector<FeatureIDType> processed_feature_ids(0);  //待处理的特征
  /*
  1.找出无效的特征点和处理过的特征点
  无效特征点：1）如果该feature的所有观测点的所有相机状态包含当前状态（即仍可被观测到），此特征点无效；
  2）feature所有的观测点个数<3，无效
  3）该feature初始化失败且checkMotion判断首尾帧距离过小，无效
  4）该feature初始化首尾帧距离足够大，但是初始化特征点的世界坐标失败，无效
  */
  for (auto iter = map_server.begin();iter != map_server.end(); ++iter) {

    auto& feature = iter->second;
    // Pass the features that are still being tracked.仍然跟踪中的特征不用与观测更新，只有跟丢的才可能进行三角话和观测更新
    if (feature.observations.find(state_server.imu_state.id) !=feature.observations.end()) 
      continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    //此步骤将所有特征进行3d计算
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } 
      else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }
    //对当前帧未观测到，但是观测次数>3且初始化成功的点，保存构建雅可比
    //所有雅可比矩阵的行数：j*(4M-3)  特征个数*（每个特征的左零空间行数）
    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  //对跟踪成功的点构建残差模型
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);  //存储所有观测到此feature的所有相机状态
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

  /*
  feature被多个相机观测到的观测模型：
  计算重投影误差并投影到左零空间
  H_xj：大小 （4M-3）*（21+6N）
  r_j:(4M-3)*1
  */
    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);
  //???类似于做误差的卡方检验？？？？？？？？？？？？？？？？？？
  //用门限测试检测基于H_xj的测量预测协方差和残差的关系是否合理  剔除外点
    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  /*
  观测更新，进行EKF，更新IMU状态、相机状态 和系统协方差矩阵
  */
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}
/*
相机状态删除：[0,n]帧中主要删除前两帧 0,1 或者n-1 n-2帧  (第1 2 帧或者倒数第3 4帧)
删除前两帧：n-2  n-3帧之间的变化角度、位置等足够大，较准确，保留相关帧  删除前面老帧
*/
void MsckfVio::findRedundantCamStates(vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(rotation*key_rotation.transpose()).angle();
    // 0.2618                                          0.4                                         0.5
    if (angle < rotation_threshold && distance < translation_threshold && tracking_rate > tracking_rate_threshold) { 
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

void MsckfVio::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)   // 20 30
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    //如果当前feature存在部分observation数据在要删除的帧中，保存对应相关的待删除的camera_id
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }
  /*
  直接删掉feature的所有观测中，与involved_cam_state_ids相关的观测：
  1）如果该特征相关的involved_cam_state_ids只包含一个相机，只需要移出该camera_id对特征的observation；
  2）involved_cam_state_ids.size()=2:
                      a.对如果未初始化且三角话失败;
                      b.未初始化  三角化成功但初始位置不正确；

  如果已经成功初始化，需要计算雅可比需要删除行数
  */
    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }
  //对于已经正确初始化、未正确初始化但三角化成功且位置正确的feature许进行雅可比构建
    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } 
      else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;
//对每个特征，计算与待删除camera_id相关的雅可比
//观测矩阵更新：对与要删除相机相关的feature 卡尔曼滤波
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    //找到要删除相机对应的列
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 21 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix. 
    /*移出相机时更新协方差矩阵操作：
    相当于将整个协方差矩阵行从底向上平移6个，列从右向左平移6个，然后再重构矩阵（行-6,列-6）完成删除操作
    */

    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
      state_server.state_cov.rows()-cam_state_end,state_server.state_cov.cols()) =
      state_server.state_cov.block(cam_state_end, 0,
      state_server.state_cov.rows()-cam_state_end,state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
      state_server.state_cov.rows(),state_server.state_cov.cols()-cam_state_end) =
      state_server.state_cov.block(0, cam_state_end,
      state_server.state_cov.rows(),state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
        state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } 
    else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;    // 8
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  printf("Start %lld online reset procedure...",++online_reset_counter);
  printf("Stardard deviation in xyz: %f, %f, %f",position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  printf("%lld online reset complete...", online_reset_counter);
  return;
}



