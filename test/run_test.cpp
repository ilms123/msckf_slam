#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

#include "System.h"

using namespace std;
using namespace cv;

std::shared_ptr<System> pSystem;

int nDelayTimes=2;
string sData_path = "/home/kiki/MyFiles/euroc_data/MH05/mav0/";
string sConfig_path = "../config/";

void PubImuData(){
   string sImu_data_file= sConfig_path + "MH_05_imu0.txt";
   ifstream fsIMU;
   fsIMU.open(sImu_data_file.c_str());
   if(!fsIMU.is_open()){
       cerr<<"failed open imu_data file ..."<<endl;
       return;
   }
   string sImu_line;
   double stimestamp;
   Eigen::Vector3d sAcc,sGyr;
   while(getline(fsIMU,sImu_line)&& !sImu_line.empty()){
       istringstream sImuData(sImu_line);
       sImuData>>stimestamp>>sGyr.x()>>sGyr.y()>>sGyr.z()>>sAcc.x()>>sAcc.y()>>sAcc.z();
       pSystem->PubImuData(stimestamp/1e9,sGyr,sAcc); // 
       usleep(5000*nDelayTimes);
   }
   fsIMU.close();
}
void PubImageData()
{
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;
	
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;
		string imagePathRight = sData_path + "cam1/data/" + sImgFileName;
		Mat img0 = imread(imagePath.c_str(), 0);
		Mat img1 = imread(imagePathRight.c_str(), 0);

		if (img0.empty()||img1.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		pSystem->PubImageData(dStampNSec / 1e9, img0,img1);
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
}

int main(int argc,char **argv){
    if(argc != 3)
	{
		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
			<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
		return -1;
	}
	sData_path = argv[1];
	sConfig_path = argv[2];
    pSystem.reset(new System(sConfig_path));

    std::thread thd_PubImuData(PubImuData);
    std::thread thd_PubImageData(PubImageData);
    std::thread the_BackEnd(&System::ProcessBackEnd,pSystem);

    return 0;
}