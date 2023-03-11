#include "System.h"
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

static std::shared_ptr<ImageProcessor> image_processor_ptr=std::shared_ptr<ImageProcessor>(new ImageProcessor());
static std::shared_ptr<MsckfVio> msckf_vio_ptr=std::shared_ptr<MsckfVio>(new MsckfVio());

System::System(const std::string& sConfig_files){
    //前端追踪
    string sConfig_file=sConfig_file+"camchain-imucam-euroc.yaml";
    image_processor_ptr->initialize(sConfig_files);
    msckf_vio_ptr->initialize(sConfig_files);

}
System::~System(){

}

void System::PubImuData(double dStampSec,const Eigen::Vector3d& sGyr,const Eigen::Vector3d& sAcc){
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
    imu_msg->timestamp=dStampSec;
    imu_msg->angular_velocity=sGyr;
    imu_msg->linear_acceleration=sAcc;
    if(dStampSec<last_imu_t){
        cerr<<"imu message is disordered ..."<<endl;
        return;
    }
    last_imu_t=dStampSec;
    m_buf.lock();
    imu_buff.push_back(imu_msg);
    m_buf.unlock();
    con.notify_one();
}

void System::PubImageData(double dStampSec,const cv::Mat& img0,const cv::Mat& img1){
    if (!init_feature)
    {
        cout << "1 PubImageData skip the first detected feature, which doesn't contain optical flow speed" << endl;
        init_feature = 1;
        return;
    }

    if (first_image_flag)
    {
        cout << "2 PubImageData first_image_flag" << endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }
    // detect unstable camera stream
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time)
    {
        cerr << "3 PubImageData image discontinue! reset the feature tracker!" << endl;
        first_image_flag = true;
        last_image_time = 0;
        return;
    }
    last_image_time = dStampSec;
    image_processor_ptr->stereoCallback(img0,img1,imu_buff,dStampSec);
    //发布feature_buff
    //!!!!添加重力未初始化判断
    if(PUB_THIS_FRAME && msckf_vio_ptr->HasInitGravityAndBias()){  //  ???初始化完成之后才发布feature信息
        std::shared_ptr<IMG_MSG> feature_points=image_processor_ptr->featureUpdateCallback(dStampSec); 
        m_buf.lock();
        feature_buff.push(feature_points);
        m_buf.unlock();
        con.notify_one();
    }
}

void System::ProcessBackEnd(){
    while(bStart_backend){   // 需要在system  init时初始化为true
        if(!msckf_vio_ptr->HasInitGravityAndBias()){
            //注！！！与unique_lock<mutex> lk(m_buf)的区别
            m_buf.lock();
            if(imu_buff.size()<200)
                continue;
            msckf_vio_ptr->initializeGravityAndBias(imu_buff);
            m_buf.unlock();   
        }
        while(feature_buff.size()>0){
            std::shared_ptr<const IMG_MSG> feature_msg_ptr=feature_buff.front();
            double cam_time=feature_msg_ptr->timestamp;
            msckf_vio_ptr->featureCallback(feature_msg_ptr,imu_buff);
            if(imu_buff.size()>0){
                m_buf.lock();
                auto it=imu_buff.begin();
                auto itend=imu_buff.end();
                while(it!=itend && (*it)->timestamp<cam_time)
                    it++;
                imu_buff.erase(imu_buff.cbegin(),it);
                m_buf.unlock();
            }
        }
 
    }
    


}