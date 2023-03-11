#include "parameters.h"
#include <opencv2/opencv.hpp>

double ROW,COL;
double gyro_bias_cov=1e-4;
double velocity_cov=0.25;
double acc_bias_cov=1e-2;
double extrinsic_rotation_cov=3.0462e-4;
double extrinsic_translation_cov=1e-4;

void readParameters(std::string config_file){
    cv::FileStorage fsSetting(config_file,cv::FileStorage::READ);


}