#include "background.cuh"
#include <time.h>
#include <cstdio>

Background::Background(Parameters &p, Data &da, Interpolation &i, Dispersion &di) : Helper(p), data(da), interp(i), disp(di)
{
	std::cout << "Background" << std::endl;

	d_bg = NULL;
	d_bg_mask = NULL;

	dimGrid_w = dim3((width - 1) / TILE_WIDTH + 1, (height_1fr - 1)/TILE_WIDTH + 1, 1);
	dimGrid_w2 = dim3((width_2x - 1) / TILE_WIDTH + 1, (height_1fr - 1)/TILE_WIDTH + 1, 1);
}

Background::~Background()
{
	if (d_bg) gpuErrchk( cudaFree(d_bg) );
	if (d_bg_mask) gpuErrchk( cudaFree(d_bg_mask) );
}

void Background::process()
{
	genBg();
	//genBgMask();
	//genBgNoise();
}

void Background::genBg()
{
	float2 *bg_frame = new float2[1 * used_width]();
	//load first ascan
	data.loadMem(p.mem_bg_ptr, 0, width, p.camStartPixel, p.camEndPixel, /*height_1fr*/1, bg_frame);
	float *bg_column_mean = new float[used_width];
	bgColumnMeanMax = 0;
	//find maximum in bg
	for (size_t i = 0; i < used_width; i++)
	{
		bg_column_mean[i] = bg_frame[i].x;
		if (bg_column_mean[i]>bgColumnMeanMax)
		{
			bgColumnMeanMax = bg_column_mean[i];
		}
	}
	gpuErrchk(cudaMalloc((void **)&d_bg, (used_width * sizeof(float))));
	gpuErrchk(cudaMemcpy(d_bg, bg_column_mean, (used_width * sizeof(float)), cudaMemcpyHostToDevice));
	for (size_t i = 0; i < used_width; i++)
	{
		bg_column_mean[i] =  1.0/(bg_column_mean[i]/ bgColumnMeanMax + p.bg_mask_correct);
	}
	gpuErrchk(cudaMalloc((void **)&d_bg_mask, (used_width * sizeof(float))));
	gpuErrchk(cudaMemcpy(d_bg_mask, bg_column_mean, (used_width * sizeof(float)), cudaMemcpyHostToDevice));

	delete[] bg_frame;
	delete[] bg_column_mean;

	std::cout << "	- Generated bg matrix" << std::endl;
}

void Background::genBgMask()
{
	//// normalize d_bg to max of 1.
	//float2 *d_bg_temp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_temp, (used_width * sizeof(float2))) );
	//gpuErrchk( cudaMemset(d_bg_temp, 0, (used_width * sizeof(float2))) );
	//divide<<<dimLine_uw,dimLine_B>>>(1, used_width, d_bg, bgColumnMeanMax, d_bg_temp); gpuErrchk( cudaPeekAtLastError() );
	//// spectral interpolation and resampling to kspace.
	//float2 *d_bg_temp_zp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_temp_zp, (width_2x * sizeof(float2))) );
	//gpuErrchk( cudaMemset(d_bg_temp_zp, 0, (width_2x * sizeof(float2))) );
	//FFT(1, used_width, d_bg_temp, d_bg_temp);
	//zero_pad<<<dimGrid_uw,dimGrid_B>>>(1, width, used_width, d_bg_temp, d_bg_temp_zp); gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaFree(d_bg_temp) );
	//IFT(1, width_2x, dimGrid_w2, dimGrid_B, d_bg_temp_zp, d_bg_temp_zp);
	//gpuErrchk(cudaMalloc((void **)&d_bg_mask, (p.batchAscans * width_2x * sizeof(float2))));
	//interp.procInterp(1, width_2x, dimLine_w2, dimLine_B, d_bg_temp_zp, nullptr);
	////print_gpu_data(1, width_2x, d_bg_temp_zp, "d_bg_temp_zp_IFT_interp");
	//gpuErrchk( cudaFree(d_bg_temp_zp) );
	//// Reciprocal is taken so we can later use it to multiply instead of divide. This is d_bg_mask.
	//reciprocal<<<dimLine_w2,dimLine_B>>>(1, width_2x, d_bg_mask); gpuErrchk( cudaPeekAtLastError() );
	//repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, width_2x, d_bg_mask); gpuErrchk( cudaPeekAtLastError() );
}

void Background::genBgNoise()
{
	//subtract<<<dimLine_uw,dimLine_B>>>(height_1fr, used_width, d_bg_frame, d_bg, d_bg_frame); gpuErrchk( cudaPeekAtLastError() );
	//FFT(height_1fr, used_width, d_bg_frame, d_bg_frame);

	////print_gpu_data(height_1fr, used_width, d_bg_frame, "FFT");

	//float2 *d_bg_frame`_zp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_frame_zp, (height_1fr * width_2x * sizeof(float2))) );
	//gpuErrchk(cudaMemset(d_bg_frame_zp, 0, (height_1fr * width_2x * sizeof(float2))));

	//
	//zero_pad<<<dimGrid_w,dimGrid_B>>>(height_1fr, width, used_width, d_bg_frame, d_bg_frame_zp); 
	//gpuErrchk( cudaPeekAtLastError() );
	/////this is the line raising the CUDA error
	//gpuErrchk( cudaFree(d_bg_frame) );														 
	//IFT(height_1fr, width_2x, dimGrid_w2, dimGrid_B, d_bg_frame_zp, d_bg_frame_zp);

	////print_gpu_data(height_1fr, width * 2, d_bg_frame_zp, "IFT");


	//float2 *d_bg_frame_interp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_frame_interp, (height_1fr * width_2x * sizeof(float2))) );
	//																
	//interp.procInterp(height_1fr, width_2x, dimLine_w2, dimLine_B, d_bg_frame_zp, d_bg_frame_interp);
	//gpuErrchk( cudaFree(d_bg_frame_zp) );

	//float2 *d_bg_noise_temp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_noise_temp, (height_1fr * width_2x * sizeof(float2))) );
	//gpuErrchk( cudaMemset(d_bg_noise_temp, 0, (height_1fr * width_2x * sizeof(float2))) );
	//
	//mult_divide<<<dimLine_w2,dimLine_B>>>(height_1fr, width_2x, data.d_gauss_win, d_bg_frame_interp, d_bg_mask, d_bg_noise_temp); gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaFree(d_bg_frame_interp) );

	//float2 *d_bg_noise_comp;
	//gpuErrchk( cudaMalloc((void **)&d_bg_noise_comp, (height_1fr * width_2x * sizeof(float2))) );

	//phi_multiply<<<dimLine_w2,dimLine_B>>>(height_1fr, width_2x, disp.d_fphi, d_bg_noise_temp, d_bg_noise_comp); gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaFree(d_bg_noise_temp) );

	//IFT(height_1fr, width_2x, dimGrid_w2, dimGrid_B, d_bg_noise_comp, d_bg_noise_comp);

	//float *d_bg_noise_temp_mag;
	//gpuErrchk( cudaMalloc((void **)&d_bg_noise_temp_mag, (height_1fr * width_2x * sizeof(float))) );
	//magnitude_db<<<dimGrid_w2,dimGrid_B>>>(height_1fr, width_2x, d_bg_noise_comp, d_bg_noise_temp_mag); gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaFree(d_bg_noise_comp) );

	//float *bg_noise_temp = new float[height_1fr * width_2x];
	//gpuErrchk( cudaMemcpy(bg_noise_temp, d_bg_noise_temp_mag, (height_1fr *  width_2x * sizeof(float)), cudaMemcpyDeviceToHost) );
	//gpuErrchk( cudaFree(d_bg_noise_temp_mag) );


	//float *bg_noise = new float[height_1fr * (used_width - 10)];
	//for (int i = 0; i < height_1fr ; i++)
	//	for (int j = 10; j < used_width; j++)
	//		bg_noise[i*(used_width-10)+j-10] = bg_noise_temp[i*used_width*2+j];
	//delete[] bg_noise_temp;


	//float *noise_lvl = new float[used_width - 10];
	//columnMean(height_1fr, (used_width - 10), bg_noise, noise_lvl);
	//delete[] bg_noise;


	//vector<float> noise_lvl_vec(used_width - 10);
	//for (int i=0; i < (used_width - 10); i++)
	//	noise_lvl_vec[i] = noise_lvl[i];
	//delete[] noise_lvl;

	//float *noise_lvl_res = new float[used_width - 10];
	//vector<float> filterA(10, 0.1f);
	//vector<float> a(1, 1.0f);

	//filtfilt(9, a, filterA, (used_width - 10), noise_lvl_vec, noise_lvl_res);


	//float *noise_bg = new float[width_trm];
	//for (int i=0; i < width_trm; i++)
	//	noise_bg[i] = noise_lvl_res[i];

	//delete[] noise_lvl_res;

	//gpuErrchk( cudaMalloc((void **)&d_bg_noise, (p.batchAscans * width_trm * sizeof(float))) );
	//gpuErrchk( cudaMemcpy(d_bg_noise, noise_bg, (width_trm * sizeof(float)), cudaMemcpyHostToDevice) );

	//repmat<<<dimLine_wt,dimLine_B>>>(p.batchAscans, width_trm, d_bg_noise); gpuErrchk( cudaPeekAtLastError() );


	//delete[] noise_bg;
}

void Background::columnMean(int h, int w, float *my_array, float *result_array)
{
	float sum = 0;
	float mean = 0;
	int count = 0;

	for (int j = 0; j < w; ++j)
	{
		for (int i = 0; i < h; ++i)
		{
			sum += my_array[i * w + j];
			count++;
		}
		mean = (float)sum / count;
		result_array[j] = mean;
		sum = 0;
		count = 0;
	}
} 

void Background::filter(int ord, std::vector<float> &a, std::vector<float> &b, int w, std::vector<float> &my_vector, std::vector<float> &result_vector)
{
	result_vector[0] = b[0] * my_vector[0];
	for (int i = 1; i < (ord + 1) ; i++)
	{
        result_vector[i] = 0.0;
        for (int j = 0; j < (i + 1); j++)
        	result_vector[i] = result_vector[i] + b[j] * my_vector[i - j];
       for (int j = 0; j < i ; j++)
        	result_vector[i] = result_vector[i] - a[j + 1] * result_vector[i - j - 1];
	}
	for (int i = (ord + 1); i < w; i++) //took off +1
	{  
		result_vector[i] = 0.0;
        for (int j = 0; j < (ord + 1); j++)
			result_vector[i] = result_vector[i] + b[j] * my_vector[i - j];
		for (int j = 0; j < ord; j++)
			result_vector[i] = result_vector[i] - a[j + 1] * result_vector[i - j - 1];
	}
}

void Background::filtfilt(int ord, std::vector<float> &a, std::vector<float> &b, int w, std::vector<float> &my_vector, float *result_array)
{
	int nfilt;

	if (b.size() > a.size())
	{
		nfilt = (int)b.size();
		for (int i = (int)a.size(); i < b.size(); i++)
			a.push_back(float(0.0));
	} 
	else
	{
		nfilt = (int)a.size();
		for (int i = (int)b.size(); i < a.size(); i++)
			b.push_back(float(0.0));
	}

	int nfact = 3 * (nfilt - 1);

	std::vector <float> my_vector_IC_begin;
	for (int i = nfact; i >= 1; i--)
		my_vector_IC_begin.push_back((float)(2.0) * my_vector[0] - my_vector[i]);

	std::vector <float> my_vector_IC_end;
	for (int i = (w - 2); i >= (w - nfact - 1); i--)
		my_vector_IC_end.push_back((float)(2.0) * my_vector[w - 1] - my_vector[i]);

	std::vector <float> my_vector_IC;
	int IC_vector_width = (int)my_vector_IC_begin.size() + (int)my_vector.size() + (int)my_vector_IC_end.size();
	my_vector_IC.reserve(IC_vector_width);
	my_vector_IC.insert(my_vector_IC.end(), my_vector_IC_begin.begin(), my_vector_IC_begin.end());
	my_vector_IC.insert(my_vector_IC.end(), my_vector.begin(), my_vector.end());
	
	my_vector_IC.insert(my_vector_IC.end(), my_vector_IC_end.begin(), my_vector_IC_end.end());

	std::vector<float> temp_result_vector(IC_vector_width);
	filter(ord, a, b, IC_vector_width, my_vector_IC, temp_result_vector);

	std::vector<float> temp_data(IC_vector_width);
	
	/* reverse the series for FILTFILT */
	for (int i = 0; i < IC_vector_width; i++)
		temp_data[i] = temp_result_vector[IC_vector_width - i - 1];

	/* do FILTER again */
	filter(ord, a, b, IC_vector_width, temp_data, temp_result_vector);

	/* reverse the series back */
	for (int i = 0; i < IC_vector_width; i++)
		temp_data[i] = temp_result_vector[IC_vector_width - i - 1];

	for (int i = nfact; i < (IC_vector_width - nfact); i++)
		result_array[i - nfact] = temp_data[i];
}