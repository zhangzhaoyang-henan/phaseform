#include "image.h"
#include <iostream> // why was this included here?

//using namespace cv;

Image::Image(int h, int w, float *im) : imageArray(im)
{
	height = w;
	width = h;
}

Image::~Image()
{
}

void Image::dspl()
{
	curImg = cv::Mat(height, width, CV_32FC1, imageArray);
	normImg(curImg);

	cv::namedWindow("Dispersion Result", CV_WINDOW_NORMAL);
	cv::resizeWindow("Dispersion Result", width/1.5, height/1.5);
	cv::moveWindow("Dispersion Result",(80+(width/1.5)),50);
	cv::destroyWindow("Gridsearch");
	cv::imshow("Dispersion Result", curImg);

	// check if esc was pressed
	char k = cv::waitKey(5000);
	if (k == 27)
	{
		cv::destroyWindow("Dispersion Result");
		cv::destroyWindow("Select an ROI with high dynamic range");
		// throw std::invalid_argument("Gridsearch Error");
	}

	cv::destroyWindow("Dispersion Result");
	cv::destroyWindow("Select an ROI with high dynamic range");
}

void Image::dsplGS()
{
	curImg = cv::Mat(height, width, CV_32FC1, imageArray);
	normImg(curImg);
	cv::normalize(curImg, curImg, 255.0, 0.0, cv::NORM_MINMAX);
	cv::Mat newImg;
	curImg.convertTo(newImg, CV_8UC1);
	cv::namedWindow("Gridsearch", CV_WINDOW_AUTOSIZE);
	cv::imshow("Gridsearch", newImg);
	cv::waitKey(1);
}

void Image::getPts()
{
	curImg = cv::Mat(height, width, CV_32FC1, imageArray);
	normImg(curImg);
	cv::namedWindow("Select an ROI with high dynamic range", CV_WINDOW_NORMAL);
	cv::resizeWindow("Select an ROI with high dynamic range", width/1.5, height/1.5);
	cv::moveWindow("Select an ROI with high dynamic range",50,50);
	cv::setMouseCallback("Select an ROI with high dynamic range", onMouse, this);

	while (1)
	{
		cv::imshow("Select an ROI with high dynamic range", curImg);
		if (ROIpts.size() == 2)
			break;
		cv::waitKey(10);
	}
}

void Image::onMouse(int event, int x, int y, int, void* points)
{
    if (event != CV_EVENT_LBUTTONUP) 
		return;
	// check for null pointer
	Image *thisImage = reinterpret_cast<Image*>(points);
	thisImage->onMouse(event, x, y);
}

void Image::onMouse(int event, int x, int y)
{
	ROIpts.push_back(cv::Point(x,y));
}

void Image::normImg(cv::Mat &image)
{
	double maxVal = 0;
	double minVal = 0;
	cv::Point minLoc = cv::Point(0);
	cv::Point maxLoc = cv::Point(0);
	cv::minMaxLoc(image, &minVal, &maxVal, &minLoc, &maxLoc);

	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++)
			image.at<float>(i, j) = (image.at<float>(i, j) - (float)minVal)/((float)maxVal - (float)minVal); 
	}
}