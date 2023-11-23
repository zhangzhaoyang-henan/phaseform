#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
namespace ULTRA_UTIL
{
	class Dom
	{
	private:
		cv::Mat im; // image filter by median
		cv::Mat edgex, edgey; // edgex: edge in x direction, edgey:...y direction, only contains can be 0 or 1 means not edge or is edge
	public:
		/* Smmoth image with ([0.5, 0, -0.5]) 1D filter
		: param image : grayscale image
		: type : cv:Mat

		: param transpose : to apply filter on vertical axis
		: type : boolean

		: param epsilon : small value to defer div by zero
		: type : float

		: return image_smoothed : smoothened image
		: type : cv::Mat */
		static cv::Mat smoothenImage(cv::Mat image, bool transpose = false, float epsilon = 1e-8);

		/* Find DOM at each pixel
		@ param src : median filtered image
		@ type : cv::Mat

		@ return domx : diff of diff on x axis
		@ type : cv::Mat

		@ return domy : diff of diff on y axis
		@ type : cv::Mat
		*/
		static void get_domXY(cv::Mat src, cv::Mat& domx, cv::Mat& domy);

		
		 /* Find contrast at each pixel
		: param Im : median filtered image
		: type : cv::Mat

		: return Cx : contrast on x axis
		: type : cv::Mat

		: return Cy : contrast on y axis
		: type : cv::Mat */
		static void contrast(cv::Mat src, cv::Mat& Cx, cv::Mat& Cy);

	private:

		/* Get Edge pixels 
		@ param image : grayscale image
		@ type : cv::Mat

		@ param edge_threshold : threshold to consider pixel as edge if its value is greater
		@ type : float

		@ assign edgex : edge pixels matrix in x - axis as 0, 1 stand for false and true
		@ type : cv::Mat

		@ assign edgey : edge pixels matrix in y - axis as 0, 1 stand for false and true
		@ type : cv::Mat */
		void edges(cv::Mat image, float edge_threshold = 0.0001);

		/* Final Sharpness Value
		:param Im : median filtered grayscale image
		: type : cv::Mat

		: param width : edge width
		: type : int

		: param sharpness_threshold : thresold to consider if a pixel is sharp
		: type : float

		: param debug : to show intermediate results
		: type : boolean

		: return S : sharpness measure(0<S<sqrt(2))
		: type : float */
		void sharpness_matrix(cv::Mat& Sx, cv::Mat& Sy, int width = 2, bool debug = false);

		/* Final Sharpness Value
		@ param Im : median filtered grayscale image
		@ type : cv::Mat

		@ param width : edge width
		@ type : int

		@ param sharpness_threshold : thresold to consider if a pixel is sharp
		@ type : float

		@ param debug : to show intermediate results
		@ type : boolean

		@ return S : sharpness measure(0<S<sqrt(2))
		@ type : float*/
		double sharpness_measure(int width, float sharpness_threshold, bool debug, double epsilon = 1e-8);

	public:
		Dom();
		~Dom();

		/*Image Sharpness Assessment
        @param img: img src or image matrix
        @type: str or cv::Mat

        @param width: text edge width
        @type: int

        @param sharpness_threshold: thresold to consider if a pixel is sharp
        @type: float

        @param edge_threshold: thresold to consider if a pixel is an edge pixel
        @type: float

        @param debug: to show intermediate results
        @type: boolean
    
        @return score: image sharpness measure(0<S<sqrt(2))
        @type: double*/
		double get_sharpness(cv::Mat src_img, int width = 2, float sharpness_threshold = 2.f, 
			float edge_threshold = 0.0001, bool debug = false, double epsilon = 1e-8);

#pragma region test
		static void unittest_smoothenImage(cv::Mat& img)
		{
			img.convertTo(img, CV_32FC1);
			img = smoothenImage(img);
			std::cout << "M = " << std::endl << " " << img << std::endl << std::endl;
		}

		void unittest_edges(cv::Mat& img)
		{
			img.convertTo(img, CV_32FC1);
			this->edges(img);
		}

		void unittest_getdomxy(cv::Mat& img)
		{
			cv::Mat Im, domx, domy;
			cv::medianBlur(img, Im, 3);
			Im.convertTo(Im, CV_64FC1);
			Im /= 255.0;
			this->get_domXY(Im, domx, domy);
			std::cout << "domx = " << std::endl << " " << domx << std::endl << std::endl;
			std::cout << "domy = " << std::endl << " " << domy << std::endl << std::endl;
		}

		void unittest_contrast(cv::Mat& img)
		{
			cv::Mat Im, Cx, Cy;
			cv::medianBlur(img, Im, 3);
			Im.convertTo(Im, CV_64FC1);
			Im /= 255.0;
			this->contrast(Im, Cx, Cy);
			std::cout << "Cx = " << std::endl << " " << Cx << std::endl << std::endl;
			std::cout << "Cy = " << std::endl << " " << Cy << std::endl << std::endl;
		}

		void unittest_sharpness_matrix(cv::Mat& img)
		{
			cv::Mat Sx, Sy;
			cv::medianBlur(img, im, 3);
			im.convertTo(im, CV_64FC1);
			im /= 255.0;
			img.convertTo(img, CV_32FC1);
			this->edges(img);
			this->sharpness_matrix(Sx, Sy);
			/*std::cout << "Sx = " << std::endl << " " << Sx << std::endl << std::endl;
			std::cout << "Sy = " << std::endl << " " << Sy << std::endl << std::endl; */
		}
#pragma endregion
	};
}

