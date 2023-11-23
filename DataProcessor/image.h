#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>

//using namespace cv;

/* 
 * OpenCV image class used mainly for dispersion.
 */

class Image
{
public:
	Image(int h, int w, float *im);
	~Image(void);

	std::vector<cv::Point> ROIpts;
	// Used for ROI selection.
	void getPts();
	// This is used during gridsearch calculation.
	void dsplGS();
	// Used to show dispersion results. The user can hit esc to reject results.
	void dspl();

private:
	int height;
	int width;
	float *imageArray;
	cv::Mat curImg;

	// Get the coordinates of mouse click when selecting the ROI.
	// http://stackoverflow.com/questions/25748404/how-to-use-cvsetmousecallback-in-class/25748777
	void onMouse(int event, int x, int y);
	static void onMouse(int event, int x, int y, int, void *pts);

public:
	// Normalize the image. This can potentially be done on the GPU to speed up gridsearch. 
	static void normImg(cv::Mat &img);
};