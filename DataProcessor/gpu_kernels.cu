#pragma once
#include "gpu_kernels.cuh"

//gpu 函数类集合
// add by tao 2018/04/15
__global__ void enface_avg(int width, int height, int depth, float* src, float* dst)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int tid = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	if (tid < width*height)
	{
		dst[tid] = 0;
		for (size_t i = tid*depth; i < (tid+1)*depth; ++i)
		{
			dst[tid] += src[i];
		}
		dst[tid] /= depth;
	}
}

__global__ void subtract(int height, int width, float2 *initial_array, float *b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width+idx].x = initial_array[i*width+idx].x - b[idx];
}

__global__ void subtract(int height, int width, uint16_t *a, float *b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width+idx].x = (float)a[i*width+idx] - b[idx];
}

//reduce the mem of bg;
__global__ void substract(int height, int width, int used_width, float2 *initial_array, float *b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idx_b = idx%used_width;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width + idx].x = initial_array[i*width + idx].x;//- b[idx_b];
}
__global__ void subtract(int height, int width, int used_width, int16_t *a, float *b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idx_b = idx%used_width;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width + idx].x = (float)a[i*width + idx];//- b[idx_b];
}
__global__ void div_subtract(int height, int width, int used_width, int16_t *a, float *b, float *c, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idx_b = idx%used_width;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width + idx].x = ((float)a[i*width + idx]- b[idx_b])*c[idx_b];
}


__global__ void multiply(int height, int width, float *a, float *b, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

    if ((w < width) && (h < height))
		result_array[index] = a[index] * b[index];
}

__global__ void divide(int height, int width, float *a, float b, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;
	
	if ((w < width) && (h < height))
		result_array[index] = a[index] / b;
}

// This one should be removed after changing code in background to use above version.
__global__ void divide(int height, int width, float *initial_array, float b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width+idx].x = initial_array[i*width+idx] / b;
}

__global__ void reciprocal(int height, int width, float2 *input_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			input_array[i*width+idx].x = 1.f/input_array[i*width+idx].x;
}

// multiply twice since the reciprocal will be passed in.
__global__ void mult_divide(int height, int width, float *a, float2 *b, float2 *c, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			result_array[i*width + idx].x = a[idx] * b[i*width + idx].x;//*c[idx].x;
}

__global__ void subt_divide(int height, int width, float *a, float *b, float c, float *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			//result_array[i*width+idx] = (a[i*width+idx] - b[i])*c;
			result_array[i*width + idx] = (a[i*width + idx] + 30)*c;
}

__global__ void subt_divide(int height, int width, float *a, float b, float c, float *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
			//result_array[i*width+idx] = (a[i*width+idx] - b[i])*c;
			result_array[i*width + idx] = (a[i*width + idx] + b)*c;
}

__global__ void phi_multiply(int height, int width, float2 *a, float2 *b, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 0; i < height; i++)
		{
			result_array[i*width+idx].x = a[idx].x * b[i*width+idx].x;
			result_array[i*width+idx].y = a[idx].y * b[i*width+idx].x;
		}
}

__global__ void combine_div_mult(int height, int width, float *da, float2 *db, float2 *dc, float2 *ma, float2* result_array)

{

	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	//int i = blockIdx.y;

	if (idx < width)

		for (int i = 0; i < height; i++)

		{
                        

			result_array[i*width+idx].x = ma[idx].x * (da[idx] * db[i*width+idx].x * dc[idx].x);

			result_array[i*width+idx].y = ma[idx].y * (da[idx] * db[i*width+idx].x * dc[idx].x);

		}



}


__global__ void d_log(int height, int width, float *initial_array, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

	if ((w < width) && (h < height))
		result_array[index] = logf(initial_array[index]);
}

__global__ void magnitude(int height, int width, float2 *initial_array, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

	if ((w < width) && (h < height))
	{	
		result_array[index] = initial_array[index].x * initial_array[index].x + initial_array[index].y * initial_array[index].y;
		result_array[index] = sqrtf(result_array[index]);
	}
}

__global__ void magnitude_db(int height, int width, float2 *initial_array, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

	if ((w < width) && (h < height))
	{	
		result_array[index] = initial_array[index].x * initial_array[index].x + initial_array[index].y * initial_array[index].y;
		result_array[index] = sqrtf(result_array[index]);
		result_array[index] = 20.0f * log10f(result_array[index]);
	}
}

__global__ void zero_pad(int height, int width, float2 *initial_array, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h*width*2 + w;
	int index_half = h*width + w;

	if ((w < ((width*0.5)-1)) && (h < height))
		result_array[index] = initial_array[index_half];
}


/*__global__ void zero_pad(int height, int width, int used_width, float2 *initial_array, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h*width*2 + w;
	//int index_half = h*width + w;
	int index_half = h*used_width + w;

	//if ((w < ((width*0.5)-1)) && (h < height))
	//if ((w < ((used_width*0.5)-1)) && (h < height))
	if ((w < ((used_width*0.5))) && (h < height))
		result_array[index] = initial_array[index_half];
}*/

//revised by @Tao 16/09/28
__global__ void zero_pad(int height, int width, int used_width, float2 *initial_array, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h*width*2 + w;
	//int index_half = h*width + w; 
	int index_half = h*used_width + w;

	//if ((w < ((width*0.5)-1)) && (h < height))
	//if ((w < ((used_width*0.5)-1)) && (h < height))
	if ((w < ((used_width*0.5))) && (h < height))
		result_array[index] = initial_array[index_half];
}

__global__ void scale_IFT(int height, int width, float scaler, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

	if ((w < width) && (h < height))
	{
		result_array[index].x *= scaler;
		result_array[index].y *= scaler;
	}
}

__global__ void scale_IFT_x(int height, int width, float scaler, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;

	if ((w < width) && (h < height))
		result_array[index].x *= scaler;
}
														
__global__ void trim_width(int height, int width, int startPixel, int endPixel, float2 *initial_array, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;
	int index_trim = h * (endPixel - startPixel + 1) + w;

	if ((w < (endPixel-startPixel+1)) && (h < height))
		result_array[index_trim] = initial_array[index+(startPixel-1)];
}

__global__ void trim_height(int height, int width, int startPixel, int endPixel, float2 *initial_array, float2 *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;
	int index_trim = (h - (startPixel - 1)) * width + w;

	if ((w < width) && (h > (startPixel - 2)) && (h < endPixel))
		result_array[index_trim] = initial_array[index];
}

__global__ void transpose(int height, int width, float *initial_array, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	int index = h * width + w;
	int transposed_index = w * height + h;

	if ((w < width) && (h < height))
		result_array[transposed_index] = initial_array[index];
}

__global__ void multiframe_transpose(int height, int width, int simult_frames, float *initial_array, float *result_array)
{
	int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((w < width) && (h < height))
		for (int j = 0; j < simult_frames; j++)
			result_array[j*width*height + w*height + h] = initial_array[j*width*height + h*width + w];
}

__global__ void repmat(int height, int width, float *input_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 1; i < height; i++)
			input_array[i*width+idx] = input_array[idx];
}

__global__ void repmat(int height, int width, float2 *input_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 1; i < height; i++)
			input_array[i*width+idx] = input_array[idx];
}

__global__ void interp_repmat(int height, int width, float *input_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width)
		for (int i = 1; i < height; i++)
			input_array[i*width+idx] = (float)i*width+input_array[idx];
}

__global__ void linear_interp(int height, int width, float *query_points, float2 *initial_array, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width )
	{
	
		int k = floorf(query_points[idx]) - 1;  //minus one since calibration file is made for matlab, which is 1 indexed.
		for (int i = 0; i < height; i++)			
			result_array[i*width+idx].x = initial_array[i*width+k].x + ((initial_array[i*width+k+1].x - initial_array[i*width+k].x) * 
										(query_points[idx] - (k+1)));
	}
}

__global__ void d_sum_elements(int height, int width, float *initial_array, float *result_array)
{    
    extern __shared__ float cache[];
    
    int w =	blockIdx.x * blockDim.x + threadIdx.x;
    int h =	blockIdx.y * blockDim.y + threadIdx.y;
    int index = h * width + w;
    int cacheIndex = threadIdx.y * blockDim.x + threadIdx.x;
    
    float temp = 0;
    
    if ((w < width) && (h < height))
    	temp += initial_array[index];
    
    cache[cacheIndex] = temp;
    __syncthreads();
    
    int i = (blockDim.x * blockDim.y) / 2;
    while (i != 0)
	{
    	if (cacheIndex < i)
    		cache[cacheIndex] += cache[cacheIndex + i];
    	__syncthreads();
    	i /= 2;
    }
    
    if (cacheIndex == 0)
    	result_array[blockIdx.y * gridDim.x + blockIdx.x] = cache[0];
}

__global__ void interp_1_1(int height, int width, float *HI, float *DD, float *RI, float2 *initial_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < width - 2)
	{
		for (int i = 0; i < height; i++)
		{
			RI[i*(width-2)+idx] = (3.0f / HI[idx+1]) * (initial_array[i*width+idx+2].x - initial_array[i*width+idx+1].x) - 
								  (3.0f / HI[idx])   * (initial_array[i*width+idx+1].x - initial_array[i*width+idx].x);
			if (idx == 0)
				RI[i*(width-2)+idx] = RI[i*(width-2)+idx] / DD[idx];
		}
	}
}

__global__ void interp_2_1(int height, int width, float *RI, float *C)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if ((idx > 0) && (idx < (width - 1)))
		for (int i = 0; i < height; i++)
			C[i*width+idx] = RI[i*(width-2)+idx-1];
}

__global__ void interp_2_2(int height, int width, float *HI, float *C)
{
	//confirm this section

	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx == 0 || idx == (width - 1))
		for (int i = 0; i < height; i++)
		{
			if (idx == 0)
				C[i*width+idx] = ((1.0f + (HI[idx] / HI[idx+1])) * C[i*width+idx+1]) - 
										 ((HI[idx] / HI[idx+1])  * C[i*width+idx+2]);
			if (idx == (width - 1))
				C[i*width+idx] = ((1.0f + (HI[idx-1] / HI[idx-2])) * C[i*width+idx-1]) - 
										 ((HI[idx-1] / HI[idx-2])  * C[i*width+idx-2]);
		}
}

__global__ void interp_2_3	(int height, int width, float *HI, float *B, float *C, float *D, float2 *initial_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < (width - 1))
	{
		for (int i = 0; i < height; i++)
		{
			D[i*width+idx] = (C[i*width+idx+1] - C[i*width+idx]) / (3.0f * HI[idx]);

			B[i*width+idx] = ((initial_array[i*width+idx+1].x - initial_array[i*width+idx].x) / HI[idx]) -
				   ((HI[idx] * (C[i*width+idx+1] + 2.0f * C[i*width+idx])) / 3.0f);
		}
	}
}

__global__ void interp_3(int height, int width, float *X, float *B, float *C, float *D,float *query_points, float2 *initial_array, float2 *result_array)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int piece = floor(query_points[idx]) - 1;
	float temp = query_points[idx] - X[piece];

	if (idx < width)
	{
	//int piece = floor(query_points[idx]) - 1;
	//float temp = query_points[idx] - X[piece];
		for (int i = 0; i < height; i++)
		{
			result_array[i*width+idx].x = initial_array[i*width+piece].x + (B[i*width+piece] * temp);
			result_array[i*width+idx].x = result_array[i*width+idx].x	 + (C[i*width+piece] * temp * temp);
			result_array[i*width+idx].x = result_array[i*width+idx].x	 + (D[i*width+piece] * temp * temp * temp);
		}
	}


}
