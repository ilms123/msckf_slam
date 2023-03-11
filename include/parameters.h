#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <Eigen/Core>
#include <memory>

//图像尺寸
extern double ROW,COL;
extern double gyro_bias_cov,velocity_cov,acc_bias_cov;
extern double extrinsic_rotation_cov,extrinsic_translation_cov;

struct IMU_MSG
{
    double timestamp;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<const IMU_MSG> ImuConstPtr;   //指向的内容不变？？

typedef struct
{
    int id; //???  
    Eigen::Vector2d uv,uvRight;
    Eigen::Vector2d uv_undist,uvRight_undist;
}FeatureUV;

struct IMG_MSG
{
    double timestamp;
    std::vector<FeatureUV> features;
};
typedef std::shared_ptr<const IMG_MSG> ImgConstPtr;

void readParameters(std::string config_file);

#endif