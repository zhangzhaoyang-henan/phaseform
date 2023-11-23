#pragma once

#include "parameters.h"
#include "gpu_kernels.cuh"
#include <string>
typedef unsigned char uchar;
#define TILE_WIDTH 16				// 2D grid parameter
#define THREADS_PER_BLOCK 256		// 1D grid parameter 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cufftErrchk(err) { cufftAssert((err), __FILE__, __LINE__); }

/* 
 * Anything used by multiple classes goes here. Functions, variables, defines.
 */

class Helper
{
public:
	Helper(Parameters &p);
	~Helper(void);

	Parameters &p;
	int used_width;
	int height_1fr, height_bfr, width, width_2x, width_trm, frames; // Common dimension and launch parameters.通用尺寸和发射参数
	dim3 dimGrid_w, dimGrid_w2, dimGrid_wt, dimGrid_B;				// Documentation can be used as reference
	dim3 dimLine_w, dimLine_w2, dimLine_wt, dimLine_B;				// for these variables

	dim3 dimGrid_uw;
	dim3 dimLine_uw, dimLine_uw2;
	float w2Recip, grayRecip;			// recipricals calculated so we can multiply instead of divide. 求出了倒数，可以乘而不是除

	// This runs on the CPU and is used in background and dispersion. 它在CPU上运行，并用于后台和分散
	void columnMean(int h, int w, float2 *my_array, float *result_array, float &columnMeanMax);

	// FFT functions that generate their own cufft plans.  产生FFT函数
	void FFT(int height, int width, float2 *initial_array, float2 *result_array);
	void IFT(int height, int width, dim3 dimGrid, dim3 dimBlock, float2 *initial_array, float2 *result_array);
	// Catch errors generated by CUDA. If Data Processor is an .exe, a breakpoint can be put here 捕获CUDA生成的错误。如果数据处理器是一个.exe，断点可以放在这里
	// and an error message will be shown in console. Macros above are used. 和一个错误消息将显示在控制台。使用了上面的宏
	static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
	static void cufftAssert(cufftResult err, const char *file, const int line);

	// Debug functions
	void output_csv(int height, int width, float *my_array, std::string flname);
	void output_csv(int height, int width, float2 *my_array, std::string flname);
	//void output_csv(int height, int width, std::vector<float> &my_array);
	//void check_function(int height, int width, float *& my_array);
	//void check_function(int height, int width, float2 *& my_array);
	//void check_function(int height, int width, std::vector<float> & my_array);

	void print_gpu_data(int height, int width, float2* gpu_data, std::string name);
	void print_gpu_data(int height, int width, uint16_t* gpu_data, std::string name);
	void print_gpu_data(int height, int width, float* gpu_data, std::string name);
	void print_cpu_data(int height, int width, float* cpu_data, std::string name);
	void print_cpu_data(int height, int width, float* cpu_data, std::string name, int id);
	void print_cpu_data(int height, int width, int16_t* cpu_data, std::string name);
	void print_cpu_data(int height, int width, uchar* cpu_data, std::string name);

private:
	static const char* cudaGetErrorEnum(cufftResult error);
};
