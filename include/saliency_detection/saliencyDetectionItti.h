//===================================================================================
// Name        : saliencyDetectionItti.h
// Author      : Oytun Akman, oytunakman@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "A Model of Saliency-Based Visual Attention
//				 for Rapid Scene Analysis" by Laurent Itti, Christof Koch and Ernst
//				 Niebur (PAMI 1998).												  
//===================================================================================

#ifndef _SALIENCYMAPITTI_H_INCLUDED_
#define _SALIENCYMAPITTI_H_INCLUDED_
// ROS
#include <ros/ros.h>
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/fill_image.h>
#include <geometry_msgs/Point.h>

// OpenCV
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

class saliencyMapItti
{
protected:
    	ros::NodeHandle nh_;
    	ros::Publisher point_pub_;
    	image_transport::ImageTransport it_;
    	image_transport::Subscriber 	image_sub_;
    	image_transport::Publisher		saliencymap_pub_;

public:
    	saliencyMapItti() : nh_("~"), it_(nh_)
    	{
    		image_sub_ = it_.subscribe("/rgbimage_in", 1, &saliencyMapItti::imageCB, this);
    		saliencymap_pub_= it_.advertise("/saliency/image", 1);
    		point_pub_ = nh_.advertise<geometry_msgs::Point>("/saliency/salientpoint", 1);
    	}


    	~saliencyMapItti()
    	{
    		nh_.shutdown();
    	}

    	void imageCB(const sensor_msgs::ImageConstPtr& msg_ptr);
    	void calculateSaliencyMap(const Mat* src, Mat* dst, int scaleBase);
    	void combineFeatureMaps(int scale);

    	Mat conspicuityMap_I;
    	Mat conspicuityMap_C;
    	Mat conspicuityMap_O;
    	Mat S;

private:
    	Mat r,g,b,R,G,B,Y,I;
    	vector<Mat> gaussianPyramid_I;
    	vector<Mat> gaussianPyramid_R;
    	vector<Mat> gaussianPyramid_G;
    	vector<Mat> gaussianPyramid_B;
    	vector<Mat> gaussianPyramid_Y;

    	void createChannels(const Mat* src);
    	void createScaleSpace(const Mat* src, vector<Mat>* dst, int scale);

    	void normalize_rgb();
    	void create_RGBY();
    	void createIntensityFeatureMaps();
    	void createColorFeatureMaps();
    	void createOrientationFeatureMaps(int orientation);
    	void mapNormalization(Mat* src);
    	void clearBuffers();

    	vector<Mat> featureMaps_I;
    	vector<Mat> featureMaps_RG;
    	vector<Mat> featureMaps_BY;
    	vector<Mat> featureMaps_0;
    	vector<Mat> featureMaps_45;
    	vector<Mat> featureMaps_90;
    	vector<Mat> featureMaps_135;
};
#endif
