//===================================================================================
// Name        : saliencyDetectionHou.cpp
// Author      : Oytun Akman, oytunakman@gmail.com
// Editor	   : Joris van de Weem, joris.vdweem@gmail.com (Conversion to ROS)
// Version     : 1.2
// Copyright   : Copyright (c) 2010 LGPL
// Description : C++ implementation of "Saliency Detection: A Spectral Residual 
//				 Approach" by Xiaodi Hou and Liqing Zhang (CVPR 2007).												  
//===================================================================================
// v1.1: Changed Gaussianblur of logamplitude to averaging blur and gaussian kernel of saliency map to sigma = 8, kernelsize = 5
//      for better consistency with the paper. (Joris)
// v1.2: Ported to Robot Operating System (ROS) (Joris)

#include <saliency_detection/saliencyDetectionHou.h>


void saliencyMapHou::imageCB(const sensor_msgs::ImageConstPtr& msg_ptr)
{
	cv_bridge::CvImagePtr cv_ptr;
	sensor_msgs::Image salmap_;
	geometry_msgs::Point salientpoint_;

	Mat image_, saliencymap_;
	Point pt_salient;
	double maxVal;

	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg_ptr, enc::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}
	cv_ptr->image.copyTo(image_);

	saliencymap_.create(image_.size(),CV_8UC1);
	saliencyMapHou::calculateSaliencyMap(&image_, &saliencymap_);

	//-- Return most salient point --//
	cv::minMaxLoc(saliencymap_,NULL,&maxVal,NULL,&pt_salient);
	salientpoint_.x = pt_salient.x;
	salientpoint_.y = pt_salient.y;


	//	CONVERT FROM CV::MAT TO ROSIMAGE FOR PUBLISHING
	saliencymap_.convertTo(saliencymap_, CV_8UC1,255);
	fillImage(salmap_, "mono8",saliencymap_.rows, saliencymap_.cols, saliencymap_.step, const_cast<uint8_t*>(saliencymap_.data));

	saliencymap_pub_.publish(salmap_);
	point_pub_.publish(salientpoint_);

	return;
}


void saliencyMapHou::calculateSaliencyMap(const Mat* src, Mat* dst)
{
	Mat grayTemp, grayDown;
	vector<Mat> mv;	
	//Size imageSize(160,120);
	Size imageSize(64,64);
	Mat realImage(imageSize,CV_64F);
	Mat imaginaryImage(imageSize,CV_64F); imaginaryImage.setTo(0);
	Mat combinedImage(imageSize,CV_64FC2);
	Mat imageDFT;	
	Mat logAmplitude;
	Mat angle(imageSize,CV_64F);
	Mat magnitude(imageSize,CV_64F);
	Mat logAmplitude_blur;
	
	cvtColor(*src, grayTemp, CV_BGR2GRAY);
	resize(grayTemp, grayDown, imageSize, 0, 0, INTER_LINEAR);
	for(int j=0; j<grayDown.rows;j++)
       	for(int i=0; i<grayDown.cols; i++)
       		realImage.at<double>(j,i) = grayDown.at<uchar>(j,i);
			
	mv.push_back(realImage);
	mv.push_back(imaginaryImage);	
	merge(mv,combinedImage);	
	dft( combinedImage, imageDFT);
	split(imageDFT, mv);	

	//-- Get magnitude and phase of frequency spectrum --//
	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	log(magnitude,logAmplitude);	
	//-- Blur log amplitude with averaging filter --//
	blur(logAmplitude, logAmplitude_blur, Size(3,3), Point(-1,-1), BORDER_DEFAULT);
	
	exp(logAmplitude - logAmplitude_blur,magnitude);
	//-- Back to cartesian frequency domain --//
	polarToCart(magnitude, angle, mv.at(0), mv.at(1), false);
	merge(mv, imageDFT);
	dft( imageDFT, combinedImage, CV_DXT_INVERSE); 
	split(combinedImage, mv);

	cartToPolar(mv.at(0), mv.at(1), magnitude, angle, false);
	GaussianBlur(magnitude, magnitude, Size(5,5), 8, 0, BORDER_DEFAULT);
	magnitude = magnitude.mul(magnitude);

	double minVal,maxVal;
	minMaxLoc(magnitude, &minVal, &maxVal);
	magnitude = magnitude / maxVal;

	Mat tempFloat(imageSize,CV_32F);
	for(int j=0; j<magnitude.rows;j++)
       	for(int i=0; i<magnitude.cols; i++)
       		tempFloat.at<float>(j,i) = magnitude.at<double>(j,i);

	resize(tempFloat, *dst, dst->size(), 0, 0, INTER_LINEAR);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "saliencymap");

	saliencyMapHou salmapHou;

	ros::spin();

	return 0;
}
