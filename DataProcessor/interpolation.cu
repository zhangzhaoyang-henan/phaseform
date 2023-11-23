#include "interpolation.cuh"
Interpolation::Interpolation(Parameters &p, Data &d) : Helper(p), data(d)
{	
	d_query_points = NULL;
	if (!p.interpLinear)  // Probably set it to NULL regardless of interp mode.
	{
		DD = NULL;
		DU = NULL;
		DL = NULL;

		d_X = NULL;
		d_HI = NULL;
		d_DD = NULL;
		d_DU = NULL;
		d_DL = NULL;
		d_RI = NULL;
		d_B = NULL;
		d_C = NULL;
		d_D = NULL;
	}
}

Interpolation::~Interpolation(void)
{
	if (d_query_points) gpuErrchk( cudaFree(d_query_points) );	// Linear

	if (!p.interpLinear)										// Spline
	{
		if (DD) delete[] DD;
		if (DU) delete[] DU;
		if (DL) delete[] DL;

		if (d_X) gpuErrchk( cudaFree(d_X) ); 
		if (d_HI) gpuErrchk( cudaFree(d_HI) );
		if (d_DD) gpuErrchk( cudaFree(d_DD) );
		if (d_DU) gpuErrchk( cudaFree(d_DU) ); 
		if (d_DL) gpuErrchk( cudaFree(d_DL) );
		if (d_RI) gpuErrchk( cudaFree(d_RI) );
		if (d_B) gpuErrchk( cudaFree(d_B) );
		if (d_C) gpuErrchk( cudaFree(d_C) );
		if (d_D) gpuErrchk( cudaFree(d_D) );
	}
}

void Interpolation::init()
{
	loadPhaseCal();

	if (!p.interpLinear)
	{
		genArrays();
		std::cout << "	- Interpolation arrays initialized" << std::endl;
	}
}

void Interpolation::loadPhaseCal()
{
	// preparing pixel_nonuniform for GPU use 
	float *pixel_nonuniform = new float[width_2x];
	data.loadFile(p.fnamePhase, width_2x, pixel_nonuniform);

	//calibration file @TaoXu
	/*std::cout<<"-----------------------------"<<std::endl;
	for(size_t i = 0; i<width_2x; ++i)
	{
		std::cout<<pixel_nonuniform[i]<<std::endl;
	}
	std::cout<<"-----------------------------"<<std::endl;*/

	gpuErrchk( cudaMalloc((void **)&d_query_points, (p.batchAscans * width_2x * sizeof(float))) );
	gpuErrchk( cudaMemcpy(d_query_points, pixel_nonuniform, (width_2x * sizeof(float)), cudaMemcpyHostToDevice) );
	interp_repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, width_2x, d_query_points); gpuErrchk( cudaPeekAtLastError() );

	delete[] pixel_nonuniform;

	std::cout << "	- Calibration file loaded" <<  std::endl;
}

void Interpolation::genArrays()
{
	X  = new float[(width_2x)*p.batchAscans],
	HI = new float[(width_2x-1)*p.batchAscans],
	DD = new float[(width_2x-2)*p.batchAscans],
	DU = new float[(width_2x-3)*p.batchAscans],
	DL = new float[(width_2x-3)*p.batchAscans];

	gpuErrchk( cudaMalloc((void **)&d_X, (width_2x * p.batchAscans * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_HI, ((width_2x - 1) * p.batchAscans * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_DD, ((width_2x - 2) * p.batchAscans * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_DU, ((width_2x - 3) * p.batchAscans * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_DL, ((width_2x - 3) * p.batchAscans * sizeof(float))) );
	
	gpuErrchk( cudaMalloc((void **)&d_RI, (height_bfr * (width_2x - 2) * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_B, (height_bfr * width_2x * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_C, (height_bfr * width_2x * sizeof(float))) );
	gpuErrchk( cudaMalloc((void **)&d_D, (height_bfr * width_2x * sizeof(float))) );

	fillArrays();

	delete[] X;
	delete[] HI;
//	delete[] DD;
//	delete[] DU;
//	delete[] DL;
}

void Interpolation::fillArrays()
{
	for (int i= 0; i < width_2x; i++)
			X[i] = i + 1.0f;
	
	for (int i = 0; i < (width_2x - 1); i++)
			HI[i] = X[i+1] - X[i];

	for (int i = 0; i < (width_2x-2); i++)
			DD[i] = 2.0f * (HI[i] + HI[i+1]);

	DD[0] = DD[0] + HI[0] + ((HI[0] * HI[0])/ HI[1]);
	DD[width_2x-3] = DD[width_2x-3] + HI[width_2x-2] + ((HI[width_2x-2] * HI[width_2x-2])/ HI[width_2x-3]);

	for (int i = 1; i < (width_2x-2); i++)
			DU[i-1] = DL[i-1] = HI[i];

	DU[0] = DU[0] - ((HI[0] * HI[0]) / HI[1]);
	DL[width_2x-4] = DL[width_2x-4] - ((HI[width_2x-2] * HI[width_2x-2]) / HI[width_2x-3]); 

	// triagonal algorithm start
	for (int i = 0; i < (width_2x-3); i++)
	{
		DU[i] = DU[i] / DD[i];
		DD[i+1] = DD[i+1] - (DL[i] * DU[i]);
	}

	h_repmat(p.batchAscans, (width_2x - 2), DD);
	h_repmat(p.batchAscans, (width_2x - 3), DU);
	h_repmat(p.batchAscans, (width_2x - 3), DL);

	gpuErrchk( cudaMemcpy(d_X, X, (width_2x * sizeof(float)), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_HI, HI, ((width_2x - 1) * sizeof(float)), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_DD, DD, ((width_2x - 2) * sizeof(float)), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_DU, DU, ((width_2x - 3) * sizeof(float)), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_DL, DL, ((width_2x - 3) * sizeof(float)), cudaMemcpyHostToDevice) );
	
	repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, width_2x, d_X); gpuErrchk( cudaPeekAtLastError() );
	repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, (width_2x - 1), d_HI); gpuErrchk( cudaPeekAtLastError() );
	repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, (width_2x - 2), d_DD); gpuErrchk( cudaPeekAtLastError() );
	repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, (width_2x - 3), d_DU); gpuErrchk( cudaPeekAtLastError() );
	repmat<<<dimLine_w2,dimLine_B>>>(p.batchAscans, (width_2x - 3), d_DL); gpuErrchk( cudaPeekAtLastError() );
}

void Interpolation::procInterp(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array)
{
	if (p.interpLinear)
		linear_interp<<<dg,db>>>(h, w, d_query_points, initial_array, result_array);
	else
		splineInterp(h, w, dg, db, initial_array, result_array);
	gpuErrchk( cudaDeviceSynchronize() );
}

void Interpolation::procInterp(int h, int w, dim3 dg, dim3 db, cudaStream_t strm, float2 *initial_array, float2 *result_array)
{
	if (p.interpLinear)
		linear_interp<<<dg,db,0,strm>>>(h, w, d_query_points, initial_array, result_array);
	else
		splineInterp(h, w, dg, db, strm, initial_array, result_array);	
}

void Interpolation::splineInterp(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array)
{	
	gpuErrchk( cudaMemset(d_RI, 0, (height_bfr * (width_2x - 2) * sizeof(float))) );
	gpuErrchk( cudaMemset(d_B, 0, (height_bfr * (width_2x) * sizeof(float))) );  
	gpuErrchk( cudaMemset(d_C, 0, (height_bfr * (width_2x) * sizeof(float))) ); 
	gpuErrchk( cudaMemset(d_D, 0, (height_bfr * (width_2x) * sizeof(float))) );
	
	doInterp(h, w, dg, db, initial_array, result_array);
}

void Interpolation::splineInterp(int h, int w, dim3 dg, dim3 db, cudaStream_t strm, float2 *initial_array, float2 *result_array)
{	
	gpuErrchk( cudaMemset(d_RI, 0, (height_bfr * (width_2x - 2) * sizeof(float))) );
	gpuErrchk( cudaMemset(d_B, 0, (height_bfr * (width_2x) * sizeof(float))) );  
	gpuErrchk( cudaMemset(d_C, 0, (height_bfr * (width_2x) * sizeof(float))) ); 
	gpuErrchk( cudaMemset(d_D, 0, (height_bfr * (width_2x) * sizeof(float))) );

	doInterp(h, w, dg, db, strm, initial_array, result_array);
}

void Interpolation::doInterp(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array)
{
	//w /= p.batchAscans;		// Don't do this here as this version is used in dispersion and background
	//h *= (p.batchAscans);		// and A-scan batching doesn't happen there.

	interp_1_1<<<dg,db>>>(h, w, d_HI, d_DD, d_RI, initial_array);
	
	gpuErrchk( cudaDeviceSynchronize() );

	float *h_ri = new float[h * (w - 2)];
	cudaMemcpy(h_ri, d_RI, (h*(w-2)*sizeof(float)),cudaMemcpyDeviceToHost);

	for (int i = 0; i < h; i++)
		for (int j = 1; j < (w-2); j++)
			h_ri[i*(w-2)+j] = (h_ri[i*(w-2)+j] - (DL[j-1] * h_ri[i*(w-2)+j-1])) / DD[j];

	for (int i = 0; i < h; i++)
		for (int j = (w-4); j >= 0; j--)
			h_ri[i*(w-2)+j] = h_ri[i*(w-2)+j] - (h_ri[i*(w-2)+j+1] * DU[j]);

	cudaMemcpy(d_RI, h_ri, (h*(w-2)*sizeof(float)),cudaMemcpyHostToDevice);
	delete[] h_ri;

	gpuErrchk( cudaDeviceSynchronize() );
	interp_2_1<<<dg,db>>>(h, w, d_RI, d_C);
	interp_2_2<<<dg,db>>>(h, w, d_HI, d_C);
	interp_2_3<<<dg,db>>>(h, w, d_HI, d_B, d_C, d_D, initial_array);
	gpuErrchk( cudaDeviceSynchronize() );
	interp_3  <<<dg,db>>>(h, w, d_X, d_B, d_C, d_D, d_query_points, initial_array, result_array);
	gpuErrchk( cudaDeviceSynchronize() );

	//w *= p.batchAscans;
	//h /= (p.batchAscans);
}

void Interpolation::doInterp(int h, int w, dim3 dg, dim3 db, cudaStream_t strm, float2 *initial_array, float2 *result_array)
{
	w /= p.batchAscans;
	h *= (p.batchAscans);

	interp_1_1<<<dg,db,0,strm>>>(h, w, d_HI, d_DD, d_RI, initial_array);
	
	gpuErrchk( cudaDeviceSynchronize() );
	float *h_ri = new float[h * (w - 2)]; //
	gpuErrchk( cudaMemcpy(h_ri, d_RI, (h*(w-2)*sizeof(float)),cudaMemcpyDeviceToHost) );

	for (int i = 0; i < h; i++)
		for (int j = 1; j < (w-2); j++)
			h_ri[i*(w-2)+j] = (h_ri[i*(w-2)+j] - (DL[j-1] * h_ri[i*(w-2)+j-1])) / DD[j];

	for (int i = 0; i < h; i++)
		for (int j = (w-4); j >= 0; j--)
			h_ri[i*(w-2)+j] = h_ri[i*(w-2)+j] - (h_ri[i*(w-2)+j+1] * DU[j]);

	gpuErrchk( cudaMemcpy(d_RI, h_ri, (h*(w-2)*sizeof(float)),cudaMemcpyHostToDevice) );
	delete[] h_ri;
	gpuErrchk( cudaDeviceSynchronize() );

	interp_2_1<<<dg,db,0,strm>>>(h, w, d_RI, d_C);
	interp_2_2<<<dg,db,0,strm>>>(h, w, d_HI, d_C);
	interp_2_3<<<dg,db,0,strm>>>(h, w, d_HI, d_B, d_C, d_D, initial_array);
	interp_3  <<<dg,db,0,strm>>>(h, w, d_X, d_B, d_C, d_D, d_query_points, initial_array, result_array);
	
	w *= p.batchAscans;
	h /= (p.batchAscans);
}

void Interpolation::h_repmat(int height, int width, float *rep_array)
{
	for (int i = 1; i < height; ++i)
		for (int j = 0; j < width; ++j)
			rep_array[i * width + j] = rep_array[j];
}

// //C++ implementation of not-a-knot spline interpolation

//float	*X =	new float[height * width_2x],
//		*HI =	new float[height * (width_2x - 1)],
//		*DD =	new float[height * (width_2x - 2)],
//		*DU =	new float[height * (width_2x - 3)],
//		*DL =	new float[height * (width_2x - 3)];
//generateInterpolationArrays(height, width_2x, X, HI, DD, DU, DL);

//void generateInterpolationArrays(int height, int width, float *x, float *hi, float *dd,  float *du, float *dl){
//
//	for (int i= 0; i < width; i++)
//			x[i] = i + 1.0f;
//	
//	for (int i = 0; i < (width - 1); i++)
//			hi[i] = x[i+1] - x[i];
//
//	for (int i = 0; i < (width-2); i++)
//			dd[i] = (float)2.0 * (hi[i] + hi[i+1]);
//
//	dd[0] = dd[0] + hi[0] + ((hi[0] * hi[0])/ hi[1]);
//	dd[width-3] = dd[width-3] + hi[width-2] + ((hi[width-2] * hi[width-2])/ hi[width-3]);
//
//	for (int i = 1; i < (width-2); i++)
//			du[i-1] = dl[i-1] = hi[i];
//
//	du[0] = du[0] - ((hi[0] * hi[0]) / hi[1]);
//	dl[width-4] = dl[width-4] - ((hi[width-2] * hi[width-2]) / hi[width-3]); 
//
//	// triagonal algorithm start
//	for (int i = 0; i < (width-3); i++){
//		du[i] = du[i] / dd[i];
//		dd[i+1] = dd[i+1] - (dl[i] * du[i]);
//	}
//
//	repmat((width),	  height, x, x);
//	repmat((width-1), height, hi, hi);
//	repmat((width-2), height, dd, dd);
//	repmat((width-3), height, du, du);
//	repmat((width-3), height, dl, dl);
//}
//
//void interp1(int height, int width, float2 *my_array, vector<float> &query_points, float *result_array){
//
//	vector<float> x(width);  //sample_points vector
//	for (int i = 0; i < width; i++)
//			x[i] = i + (float)1.0;
//	
//	vector<float> hi(width-1);
//	for (int i = 0; i < (width-1); i++)
//			hi[i] = x[i+1] - x[i];
//
//	vector<float> dd(width-2);
//	vector<float> ri(width-2);
//	vector<float> du(width-3);
//	vector<float> dl(width-3);
//	vector<float> b(width, 0.0);
//	vector<float> c(width, 0.0);
//	vector<float> d(width, 0.0);
//
//	for (int h = 0; h < height; h++){
//
//		for (int i = 0; i < (width-2); i++){
//			dd[i] = 2.0f * (hi[i] + hi[i+1]);
//			ri[i] = (3.0f / hi[i+1]) * (my_array[h*width+i+2].x - my_array[h * width + i + 1].x) - 
//					(3.0f / hi[i]) * (my_array[h * width + i + 1].x - my_array[h * width + i].x);
//		}
//
//		dd[0] = dd[0] + hi[0] + ((hi[0] * hi[0])/ hi[1]);
//		dd[width-3] = dd[width-3] + hi[width-2] + ((hi[width-2] * hi[width-2])/ hi[width-3]);
//
//		for (int i = 1; i < (width-2); i++)
//			du[i-1] = dl[i-1] = hi[i];
//
//		du[0] = du[0] - ((hi[0] * hi[0]) / hi[1]);
//		dl[width-4] = dl[width-4] - ((hi[width-2] * hi[width-2]) / hi[width-3]); 
//
//		// triagonal algorithm
//		//m = size - 2;
//		for (int i = 0; i < (width-3); i++){
//			du[i] = du[i] / dd[i];
//			dd[i+1] = dd[i+1] - (dl[i] * du[i]);
//		}
//
//		ri[0] = ri[0] / dd[0];
//
//		for (int i = 1; i < (width-2); i++)
//			ri[i] = (ri[i] - (dl[i-1] * ri[i-1])) / dd[i];
//	
//		for (int i = (width-4); i >= 0; i--)
//			ri[i] = ri[i] - (ri[i+1] * du[i]);
//		// end triagonal
//
//		for (int i = 1; i < (width-1); i++)
//			c[i] = ri[i-1];
//
//		c[0] = ((1.0f + (hi[0] / hi[1])) * c[1]) - ((hi[0] / hi[1]) * c[2]);
//		c[width-1] = ((1.0f + (hi[width-2] / hi[width-3])) * c[width-2]) - ((hi[width-2] / hi[width-3]) * c[width-3]);
//
//		for (int i = 0; i < (width-1); i++){
//			d[i] = (c[i+1] - c[i]) / (3.0f * hi[i]);
//			b[i] = ((my_array[h * width + i + 1].x - my_array[h * width + i].x) / hi[i]) - ((hi[i] * (c[i+1] + 2.0f * c[i])) / 3.0f);
//		}
//
//		for (int i = 0; i < width; ++i){
//			int piece = (int)floor(query_points[i]) - 1;
//			float temp = query_points[i] - x[piece];  //there may be a discrepency on the last iteration
//		
//			result_array[h * width + i] = my_array[h * width + piece].x + (b[piece] * temp);
//			result_array[h * width + i] = result_array[h * width + i]	+ (c[piece] * temp * temp);
//			result_array[h * width + i] = result_array[h * width + i]	+ (d[piece] * temp * temp * temp);
//		}
//	}
//}
