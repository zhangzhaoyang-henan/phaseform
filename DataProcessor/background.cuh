#pragma once

#include "dispersion.cuh"

/* 
 * Background calculates the DC level in the image.
 */

class Background : public Helper
{
public:
	Background(Parameters &p, Data &da, Interpolation &i, Dispersion &di);
	~Background(void);
	
	float *d_bg, *d_bg_mask;		// temp bg_mask.
	/*float *bg_column_mean;*/
	//float2 *d_bg_mask;		

	void process();
	
private:
	Data &data;
	Interpolation &interp;
	Dispersion &disp;

	float bgColumnMeanMax;
	//float2 *d_bg_frame;

	void genBg();
	void genBgMask();
	void genBgNoise();

	// This version of columnMean doesn't return a column mean max result. 
	void columnMean(int h, int w, float *my_array, float *result_array);

	// filter and filtfilt are based on http://mechatronics.ece.usu.edu/yqchen/filter.c/
	void filter(int ord, std::vector<float> &a, std::vector<float> &b, int w, std::vector<float> &my_vector, std::vector<float> &result_vector);
	void filtfilt(int ord, std::vector<float> &a, std::vector<float> &b, int w, std::vector<float> &my_vector, float *result_array);
};