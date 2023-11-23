#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdint.h>

/* 
 * All GPU functions are here.
 * 
 * To Do: Fix Spline interpolation
 */

//enface average
__global__ void enface_avg(int width, int height, int depth, float* src, float* dst);

__global__ void subtract(int height, int width, float2 *initial_array, float *b, float2 *result_array);
__global__ void subtract(int height, int width, uint16_t *a, float *b, float2 *result_array);
__global__ void substract(int height, int width, int used_width, float2 *initial_array, float *b, float2 *result_array);
__global__ void subtract(int height, int width, int used_width, int16_t *a, float *b, float2 *result_array);
__global__ void div_subtract(int height, int width, int used_width, int16_t *a, float *b, float *c, float2 *result_array);
__global__ void multiply(int height, int width, float *a, float *b, float *result_array);
__global__ void divide(int height, int width, float *a, float b, float *result_array);
__global__ void divide(int height, int width, float *initial_array, float b, float2 *result_array);
__global__ void reciprocal(int height, int width, float2 *input_array);
__global__ void mult_divide(int height, int width, float *a, float2 *b, float2 *c, float2 *result_array);
__global__ void subt_divide(int height, int width, float *a, float *b, float c, float *result_array);
__global__ void subt_divide(int height, int width, float *a, float b, float c, float *result_array);

// apply dispersion compensation
__global__ void phi_multiply(int height, int width, float2 *a, float2 *b, float2 *result_array);
__global__ void combine_div_mult(int height, int width, float *da, float2 *db, float2 *dc, float2 *ma, float2* result_array);
__global__ void d_log(int height, int width, float *initial_array, float *result_array);
__global__ void magnitude(int height, int width, float2 *initial_array, float *result_array);
__global__ void magnitude_db(int height, int width, float2 *initial_array, float *result_array);
__global__ void zero_pad(int height, int width, float2 *initial_array, float2 *result_array);
__global__ void zero_pad(int height, int width, int used_width, float2 *initial_array, float2 *result_array);
__global__ void scale_IFT(int height, int width, float scaler, float2 *result_array);
__global__ void scale_IFT_x(int height, int width, float scaler, float2 *result_array);

// Cropping functions. Trim height is only used for ROI in dispersion.
__global__ void trim_width(int height, int width, int startPixel, int endPixel, float2 *initial_array, float2 *result_array);
__global__ void trim_height(int height, int width, int startPixel, int endPixel, float2 *initial_array, float2 *result_array);

// These should be combinable
__global__ void transpose(int height, int width, float *initial_array, float *result_array);
__global__ void multiframe_transpose(int height, int width, int simult_frames, float *initial_array, float *result_array);

// Repeats an A-scan of an array to fit p.batchAscans dimensions. Named after MATLAB function.
__global__ void repmat(int height, int width, float *input_array);
__global__ void repmat(int height, int width, float2 *input_array);
// Used for pixel_nonuniform repmat.
__global__ void interp_repmat(int height, int width, float *input_array);

__global__ void linear_interp(int height, int width, float *query_points, float2 *initial_array, float2 *result_array);
__global__ void d_sum_elements(int height, int width, float *initial_array, float *result_array);

// Cubic spline interpolation functions
__global__ void interp_1_1(int height, int width,float *HI, float *DD, float *RI, float2 *initial_array);
//__global__ void interp_1_2	// these 2 need to be implemented
//__global__ void interp_1_3	//
__global__ void interp_2_1(int height, int width, float *RI, float *C);
__global__ void interp_2_2(int height, int width, float *HI, float *C);
__global__ void interp_2_3(int height, int width, float *HI, float *B, float *C, float *D, float2 *initial_array);
__global__ void interp_3(int height, int width, float *X, float *B, float *C, float *D,float *query_points, float2 *initial_array, float2 *result_array);
