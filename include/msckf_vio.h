#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <queue>


#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include "parameters.h"

class MsckfVio {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MsckfVio();
    MsckfVio(const MsckfVio&) = delete;
    MsckfVio operator=(const MsckfVio&) = delete;
    ~MsckfVio() {}

    bool initialize(const std::string& sConfig_files);
    bool HasInitGravityAndBias();
    void initializeGravityAndBias(std::vector<ImuConstPtr>& imu_buff);
    void featureCallback(const ImgConstPtr img_msg,const std::vector<ImuConstPtr>& imu_buff);

    void reset();

    typedef boost::shared_ptr<MsckfVio> Ptr;
    typedef boost::shared_ptr<const MsckfVio> ConstPtr;

  private:
    //StateServer Store one IMU states and several camera states for constructing measurement model.
    struct StateServer {
      IMUState imu_state;
      CamStateServer cam_states;
      // State covariance matrix
      Eigen::MatrixXd state_cov;
      Eigen::Matrix<double, 12, 12> continuous_noise_cov;   //相当于x=F*X+G*n  中的N,噪声项(ng,bg,na,ba)
    };

    bool loadParameters(const std::string& sConfig_files);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(const double& time_bound,const std::vector<ImuConstPtr>& imu_buff);
    void processModel(const double& time,const Eigen::Vector3d& m_gyro,const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,const Eigen::Vector3d& gyro,const Eigen::Vector3d& acc);

    // Measurement update
    void stateAugmentation(const double& time);
    void addFeatureObservations(const ImgConstPtr img_msg_ptr);
    void measurementJacobian(const StateIDType& cam_state_id,const FeatureIDType& feature_id,
        Eigen::Matrix<double, 4, 6>& H_x,Eigen::Matrix<double, 4, 3>& H_f,Eigen::Vector4d& r);

    void featureJacobian(const FeatureIDType& feature_id,const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);

    void measurementUpdate(const Eigen::MatrixXd& H,const Eigen::VectorXd& r);
    bool gatingTest(const Eigen::MatrixXd& H,const Eigen::VectorXd&r, const int& dof);
    void removeLostFeatures();
    void findRedundantCamStates(std::vector<StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer();
    void onlineReset();
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    MapServer map_server;

    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine when to reset the system online. 
    // Otherwise, the ever-increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning. Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Framte rate of the stereo images. This variable is only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;
};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;



#endif
