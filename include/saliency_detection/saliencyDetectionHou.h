//===================================================================================
// Name        : saliencyDetectionHou.h
// Author      : Oytun Akman, oytunakman@gmail.com
// Version     : 1.0
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "Saliency Detection: A Spectral Residual 
//				 Approach" by Xiaodi Hou and Liqing Zhang (CVPR 2007).												  
//===================================================================================

#ifndef _SALIENCYMAPHOU_H_INCLUDED_
#define _SALIENCYMAPHOU_H_INCLUDED_

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

class saliencyMapHou
{
protected:
    	ros::NodeHandle nh_;
    	ros::Publisher point_pub_;
    	image_transport::ImageTransport it_;
    	image_transport::Subscriber 	image_sub_;
    	image_transport::Publisher		saliencymap_pub_;

public:
    	saliencyMapHou() : nh_("~"), it_(nh_)
    	{
    		image_sub_ = it_.subscribe("/rgbimage_in", 1, &saliencyMapHou::imageCB, this);
    		saliencymap_pub_= it_.advertise("/saliency/image", 1);
    		point_pub_ = nh_.advertise<geometry_msgs::Point>("/saliency/salientpoint", 1);
    	}


    	~saliencyMapHou()
    	{
    		nh_.shutdown();
    	}

    	void imageCB(const sensor_msgs::ImageConstPtr& msg_ptr);
    	void calculateSaliencyMap(const Mat* src, Mat* dst);
};
#endif
