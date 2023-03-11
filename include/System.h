#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <fstream>
#include <condition_variable>
#include <eigen3/Eigen/Core>

#include "image_processor.h"
#include "parameters.h"
#include "msckf_vio.h"


class System
{
public:
    System(const std::string& sConfig_files);
    ~System();
    void PubImuData(double dStampSec,const Eigen::Vector3d& sGyr,const Eigen::Vector3d& sAcc);
    void PubImageData(double dStampSec,const cv::Mat& img0,const cv::Mat& img1);
    void ProcessBackEnd();
private:
    bool bStart_backend=true;  
    // imu
    std::vector<ImuConstPtr> imu_buff;
    double last_imu_t=0;
    std::mutex m_buf;
    std::condition_variable con;
    //img-feature
    std::queue<ImgConstPtr> feature_buff;
    bool init_feature=0;
    bool first_image_flag=true;
    double last_image_time=0;
    double first_image_time;
    bool PUB_THIS_FRAME;  // true


};

#endif