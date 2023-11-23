#include "Dom.h"
#include <algorithm>
using namespace ULTRA_UTIL;

Dom::Dom()
{
}

Dom::~Dom()
{
}

cv::Mat Dom::smoothenImage(cv::Mat image, bool transpose, float epsilon)
{
	// smoothing filter
	cv::Mat filter((cv::Mat_< float >(3, 3) <<
		0, 0, 0, 
		-0.5, 0, 0.5,
		0, 0, 0));
	// change image axis for column convolution
	if (transpose)
	{
		cv::transpose(image, image);
	}

	// Convolve grayscale image with smoothing filter
	// nc::NdArray<float> image_smoothed = nc::filter::convolve1d(image, weights, nc::filter::Boundary::CONSTANT, 0.f);
	cv::Mat image_smoothed;
	cv::filter2D(image, image_smoothed, -1, filter, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_CONSTANT);

	//change image axis after column convolution
	if (transpose)
	{
		cv::transpose(image_smoothed, image_smoothed);
	}

	double minVal = 0;
	double maxVal = 0;
	cv::Point minLoc(0,0);
	cv::Point maxLoc(0,0);

	cv::minMaxLoc(image_smoothed, &minVal, &maxVal, &minLoc, &maxLoc);

	// Normalize smoothened grayscale image
	image_smoothed = cv::abs(image_smoothed) / (maxVal + epsilon);
	return image_smoothed;
}

void Dom::get_domXY(cv::Mat src, cv::Mat& domx, cv::Mat& domy)
{
	//padding and then shift up
	//copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
	cv::Mat median_shit_up = cv::Mat::zeros(src.size(), src.type());
	cv::Mat median_shit_down = cv::Mat::zeros(src.size(), src.type());
	//shift up filled with zero
	src(cv::Rect(0, 2, src.cols, src.rows - 2)).copyTo(median_shit_up(cv::Rect(0, 0, 
		median_shit_up.cols, median_shit_up.rows - 2)));
	//shift down filled with zero
	src(cv::Rect(0, 0, src.cols, src.rows - 2)).copyTo(median_shit_down(cv::Rect(0, 2,
		median_shit_down.cols, median_shit_down.rows - 2)));
	domx = cv::abs(median_shit_up - src * 2 + median_shit_down);

	cv::Mat median_shit_left = cv::Mat::zeros(src.size(), src.type());
	cv::Mat median_shit_right = cv::Mat::zeros(src.size(), src.type());
	//shift left filled with zero
	src(cv::Rect(2, 0, src.cols-2, src.rows)).copyTo(median_shit_left(cv::Rect(0, 0,
		median_shit_left.cols-2, median_shit_left.rows )));
	//shift right filled with zero
	src(cv::Rect(0, 0, src.cols-2, src.rows)).copyTo(median_shit_right(cv::Rect(2, 0,
		median_shit_right.cols-2, median_shit_right.rows)));
	domy = cv::abs(median_shit_left - src * 2 + median_shit_right);
}

void Dom::contrast(cv::Mat src, cv::Mat& Cx, cv::Mat& Cy)
{
	cv::Mat mat_shift_down = cv::Mat::zeros(src.size(), src.type());
	cv::Mat mat_shift_right = cv::Mat::zeros(src.size(), src.type());

	// shift down by one pixel
	src(cv::Rect(0, 0, src.cols, src.rows - 1)).copyTo(mat_shift_down(cv::Rect(0, 1,
		mat_shift_down.cols, mat_shift_down.rows - 1)));
	Cx = cv::abs(src- mat_shift_down);

	// shift right by one pixel
	src(cv::Rect(0, 0, src.cols - 1, src.rows)).copyTo(mat_shift_right(cv::Rect(1, 0,
		mat_shift_right.cols - 1, mat_shift_right.rows)));
	Cy = cv::abs(src - mat_shift_right);
}

void Dom::edges(cv::Mat image, float edge_threshold)
{
	// smooth image along x-axis£¬need transpose in c++ 
	cv::Mat img_x = smoothenImage(image, true);
	// smooth image along x-axis£¬need transpose in c++
	cv::Mat img_y = smoothenImage(image, false);
	this->edgex = (img_x > edge_threshold) / 255;
	this->edgex.convertTo(this->edgex, CV_64FC1);
	//std::cout << "edgex = " << std::endl << " " << this->edgey << std::endl << std::endl;
	this->edgey = (img_y > edge_threshold) / 255;
	this->edgey.convertTo(this->edgey, CV_64FC1);
	//std::cout << "edgey = " << std::endl << " " << this->edgey << std::endl << std::endl;
}

void Dom::sharpness_matrix(cv::Mat& Sx, cv::Mat& Sy, int width, bool debug)
{
	// Compute dod measure on both axis
	cv::Mat domx, domy;
	this->get_domXY(this->im, domx, domy);

	// Compute sharpness
	cv::Mat Cx, Cy;
	this->contrast(this->im, Cx, Cy);

	// Filter out contrast at pixels other than edges
	Cx = Cx.mul(this->edgex);
	Cy = Cy.mul(this->edgey);

	/*std::cout << "Cx = " << std::endl << " " << Cx << std::endl << std::endl;
	std::cout << "Cy = " << std::endl << " " << Cy << std::endl << std::endl;*/

	// initialize sharpness matriz with 0's
	Sx = cv::Mat::zeros(domx.size(), CV_64FC1);
	Sy = cv::Mat::zeros(domy.size(), CV_64FC1);

	/*std::cout << "domx = " << std::endl << " " << domx << std::endl << std::endl;
	std::cout << "domy = " << std::endl << " " << domy << std::endl << std::endl;*/

	// Compute Sx
	cv::Mat rsum;
	cv::Mat dn;
	for (size_t i = width; i < domx.rows - width; i++)
	{
		cv::reduce(cv::abs(domx.rowRange(i - width, i + width)), rsum, 0, cv::REDUCE_SUM);
		//rsum = cv::abs(rsum);
		cv::reduce(Cx.rowRange(i - width, i + width), dn, 0, cv::REDUCE_SUM);
		for (size_t j = 0; j < Sx.cols; j++)
		{
			if (dn.at<double>(0, j) > 1e-3)
			{
				Sx.at<double>(i, j) = rsum.at<double>(0, j) / dn.at<double>(0, j);
			}
			else
			{
				Sx.at<double>(i, j) = 0.0;
			}
		}
	}

	// Compute Sy
	cv::Mat csumy;
	cv::Mat dny;
	for (size_t i = width; i < domy.cols - width; i++)
	{
		cv::reduce(cv::abs(domy.colRange(i - width, i + width)), csumy, 1, cv::REDUCE_SUM);
		csumy = cv::abs(csumy);
		cv::reduce(Cy.colRange(i - width, i + width), dny, 1, cv::REDUCE_SUM);
		for (size_t j = 0; j < Sy.rows; j++)
		{
			if (dny.at<double>(j, 0) > 1e-3)
			{
				Sy.at<double>(j, i) = csumy.at<double>(j, 0) / dny.at<double>(j, 0);
			}
			else
			{
				Sy.at<double>(j, i) = 0.0;
			}
		}
	}

}

double Dom::sharpness_measure(int width, float sharpness_threshold, bool debug, double epsilon)
{
	cv::Mat Sx, Sy;
	this->sharpness_matrix(Sx, Sy, width, debug);

	// Filter out sharpness at pixels other than edges
	Sx = Sx.mul(this->edgex);
	Sy = Sy.mul(this->edgey);

	int n_sharpx = cv::countNonZero(Sx >= sharpness_threshold);
	int n_sharpy = cv::countNonZero(Sy >= sharpness_threshold);

	int n_edgex = cv::countNonZero(this->edgex);
	int n_edgey = cv::countNonZero(this->edgey);

	double Rx = n_sharpx *1.0 / (n_edgex + epsilon);
	double Ry = n_sharpy *1.0 / (n_edgey + epsilon);

	double S = std::sqrt(Rx*Rx + Ry*Ry);

	if (debug)
	{
		std::cout << "Sharpness: " << S << std::endl;
		std::cout << "Rx: " << Rx << "Ry: " << Ry << std::endl;
		std::cout << "Sharpx: " << n_sharpx<<", Sharpy: " << n_sharpy 
			<< ", Edges :" << n_edgex << "," << n_edgey << std::endl;
	}

	return S;
}

double Dom::get_sharpness(cv::Mat src_img, int width,
	float sharpness_threshold, float edge_threshold, bool debug, double epsilon)
{
	// median filter src image
	cv::medianBlur(src_img, this->im, 3);
	this->im.convertTo(this->im, CV_64FC1);
	im /= 255.0;
	
	// Initialize edge(x | y) matrices
	src_img.convertTo(src_img, CV_32FC1);
	this->edges(src_img, edge_threshold);

	double score = this->sharpness_measure(width, sharpness_threshold, debug, epsilon);
	return score;
}