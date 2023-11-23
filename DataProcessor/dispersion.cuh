#pragma once

#include <vector>

#include "interpolation.cuh"
#include "image.h"
#include "Dom.h"

/* 
 * The dispersion compensation class. Manual or gridsearch modes.
 */

class Dispersion : public Helper
{
public:
	Dispersion(Parameters &p, Data &d, Interpolation &i);
	~Dispersion(void);

	float2 *d_fphi;						// Output of the class.

	void process();
	void process(int dispModeGS, float &a2, float &a3);
private:
	Data &data;
	Interpolation &interp;
													// Some dispersion specific dimensions and launch parameters
	int width_h, width_ROI, height_ROI;				// for half a-scan length, and ROI. 'wf' is for full width, but
	dim3 dimGrid_wh, dimGrid_wf, dimGrid_wROI;		// ROI height, and wROI uses both ROI height & width.
	dim3 dimGrid_uw;
	std::vector<float> dispParam, a2;					// dispParam holds A2, A3, 0, 0. 
	float2 *fphi, *d_fringe_interp, *d_fringe_frame;
	std::vector<int> ROIbounds, w;

	cufftHandle plan_ROIw;
	
	//DOM calculator
	ULTRA_UTIL::Dom m_dom;

	void dispModeMan();								// Use user entered dispersion values (manual entry mode).
	void getROI();									// Get region to analyze from user.
	void gridsearch();								// Top level gridsearch function. Calls 2nd and 3rd order.
	void secondOrder();								// These two call getParam.
	void thirdOrder();								// 

	/*
	* method 1: top level two dimension grid search function
	*/
	void twoDOrderSearch();
	
	/*
	* method 2: recursive two dimension grid search for A2 and A3
	*/
	void recursiveGridSearch();

	void getParam(int it, std::vector<float> &result_vector); // Actual gridsearch calculation.
	void genFphi();									// Once we have a2 and a3 calculated, this generates d_fphi.
	void previewComp();								// Preview result.

	// linSpace attempts to replicate MATLAB linspace functionality.
	void linSpace(float min, float max, int pts, std::vector<float> &result_vector);
	// sums all of the elements in the array on the GPU. 
	void sumElements(int height, int width, float *initial_array, float &sum);
};