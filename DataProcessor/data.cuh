#pragma once

#include <fstream>
#include <cerrno>

#include "helper.cuh"

/*
* Data has 2 purposes: create the gaussian window array, and load the calibration, background, and data binary files into arrays.
*
* To Do: Retest various file reading methods for speed.
*/

class Data : public Helper
{
public:
	Data(Parameters &p);
	~Data(void);

	float *d_gauss_win;			// Output of class.

	//first version is used for calibration file, second is for background and disperion, and last is for the data file.
	//these probably can be combined 
	void loadFile(std::string fname, int length, float *result_array);
	//void loadFile(std::string fname, int startLocation, int length, float2 *result_array);
	//void loadFile(std::string fname, int startLocation, int length, uint16_t *result_array);

	// for need: pixels 4096, just the 1--4000
	// add by Brian @mic515.lehigh.edu
	//void loadFile(std::string fname, int startLocation, int all_pixels, int used_pixels, int height, float2 *result_array);
	void loadFile(std::string fname, int startLocation, int all_pixels, int used_pixels, int height, uint16_t *result_array);
	//void loadFile(std::string fname, long long startLocation, int all_pixels, int used_pixels, int height, float2 *result_array);
	void loadFile(std::string fname, long long startLocation, int all_pixels, int used_pixels, int height, uint16_t *result_array);

	//for need: pixels 4096, just the 1--4000, and height*n like(600,1200)
	void loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int used_pixels, int height, float2 *result_array);


	// version for start pixel and end pixel, load data from disk file 
	void loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, uint16_t *result_array);
	void loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, float2 *result_array);

	//version of process the memory data
	void loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, float2 *result_array);
	void loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, uint16_t *result_array);
	void loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, int16_t *result_array);
private:
	void genGaussWin();
};