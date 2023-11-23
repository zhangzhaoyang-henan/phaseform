#pragma once

#include "data.cuh"


/* 
 * Interpolation gets its own class because of the large number of support arrays necessary for spline interpolation.
 * 
 * Spline interpolation based on matlab code found at http://www.pcs.cnu.edu/~bbradie/matlab.html
 */


class Interpolation : public Helper
{
public:
	Interpolation(Parameters &p, Data &d);
	~Interpolation(void);
	
	void init(); // Generate support arrays if necessary.
	// 2 main functions are available, one takes a cudaStream_t parameter.
	void procInterp(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array);
	void procInterp(int h, int w, dim3 dg, dim3 db, cudaStream_t strm, float2 *initial_array, float2 *result_array);

private:
	Data &data;
	
	float *d_query_points; // This array is only used by this class. This is 'pixel_nonuniform.'

	// cubic spline support arrays
	float *X, *HI, *DD, *DU, *DL;
	float *d_X, *d_HI, *d_DD, *d_DU, *d_DL;
	float *d_RI, *d_B, *d_C, *d_D;

	 
	void loadPhaseCal();	// Initialization functions.
	void genArrays();		//
	void fillArrays();		//

	// All of these are for spline interpolation.
	void splineInterp(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array);
	void splineInterp(int h, int w, dim3 dg, dim3 db, cudaStream_t strm, float2 *initial_array, float2 *result_array);
	void doInterp(int height, int width, dim3 dimGrid, dim3 dimBlock, float2 *initial_array, float2 *result_array);
	void doInterp(int height, int width, dim3 dimGrid, dim3 dimBlock, cudaStream_t strm, float2 *initial_array, float2 *result_array);
	void h_repmat(int vector_size, int timesToRepeat, float *rep_array);
};