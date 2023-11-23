#include "dispersion.cuh"
#include <iostream>
using namespace std;

Dispersion::Dispersion(Parameters &p, Data &d, Interpolation &i) : Helper(p), data(d), interp(i)
{
	cout << "Dispersion" << endl;
	d_fphi = NULL;

	dispParam.resize(4);

	dimGrid_w = dim3((width - 1) / TILE_WIDTH + 1, (height_1fr - 1) / TILE_WIDTH + 1, 1);
	dimGrid_w2 = dim3((width_2x - 1) / TILE_WIDTH + 1, (height_1fr - 1) / TILE_WIDTH + 1, 1);
	dimGrid_wt = dim3((width_trm - 1) / TILE_WIDTH + 1, (height_1fr - 1) / TILE_WIDTH + 1, 1);

	dimGrid_uw = dim3((used_width - 1) / TILE_WIDTH + 1, (height_1fr - 1) / TILE_WIDTH + 1, 1);

	// half A-scan length parameters for ROI selection.
	width_h = p.numCameraPixels / 3 * 2;
	dimGrid_wh = dim3((width_h - 1) / TILE_WIDTH + 1, (height_1fr - 1) / TILE_WIDTH + 1, 1);
}

Dispersion::~Dispersion()
{
	if (d_fphi) gpuErrchk(cudaFree(d_fphi));
}

void Dispersion::process()
{
	gpuErrchk(cudaMalloc((void **)&d_fphi, (p.batchAscans*width_2x * sizeof(float2))));

	if (p.dispModeGS)
	{
		getROI();
		gridsearch();
		// previewComp();
	}
	else
		dispModeMan();
}

void Dispersion::process(int dispModeGS, float &a2, float &a3)
{
	gpuErrchk(cudaMalloc((void **)&d_fphi, (p.batchAscans*width_2x * sizeof(float2))));

	if (dispModeGS)
	{
		getROI();
		gridsearch();
		previewComp();
		// return calculation result of dispersion params
		a2 = dispParam[0];
		a3 = dispParam[1];
	}
	else
	{
		dispModeMan();
	}

}


void Dispersion::dispModeMan()
{
	fphi = new float2[width_2x];
	vector<float> phi(width_2x, 0.0);

	//test the mannul dispersion bug @TaoXu

	/*std::ofstream logfile("diperstion_optimization.txt",std::ofstream::app);
	logfile << "dispersion A2: "<<p.dispA2<<";	dispersion A3:"<<p.dispA3<<endl;*/

	dispParam[0] = p.dispA2;
	dispParam[1] = p.dispA3;
	dispParam[2] = dispParam[3] = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < width_2x; j++)
			//phi[j] = phi[j] + dispParam[i] * powf((j+1.f), (i+2));
			phi[j] = phi[j] + dispParam[i] * powf((j - width + 1.f), (i + 2));

	// debug for the manual dispersion @TaoXu
	/*for(size_t i = 0; i < width_2x; ++i)
	{
		logfile<<phi[i]<<"	";
	}
	logfile<<endl;*/
	//logfile.close();

	double temp = 0;
	for (int i = 0; i < width_2x; i++)
	{
		temp = -phi[i] / width_2x;
		fphi[i].x = cos(temp);
		fphi[i].y = sin(temp);
	}
	gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));

	repmat << <dimLine_w2, dimLine_B >> > (p.batchAscans, width_2x, d_fphi); gpuErrchk(cudaPeekAtLastError());
	delete[] fphi;
}

void Dispersion::getROI()
{
	unsigned long long startposition = 0;
	//float2 *initial_frame = new float2[height_1fr * width]();
	float2 *initial_frame = new float2[height_1fr * used_width]();
	//data.loadFile(p.fnameData, 0, (height_1fr*width), initial_frame);
	//data.loadFile(p.fnameData, startposition, width, used_width, height_1fr, initial_frame);
	//data.loadFile(p.fnameData, startposition, width, p.camStartPixel, p.camEndPixel, height_1fr, initial_frame);
	//data.loadMem(p.mem_data_ptr, startposition, width, p.camStartPixel, p.camEndPixel, height_1fr, initial_frame);
	//add by tao @06/28/2017
	/*data.loadMem(p.mem_bg_ptr, startposition, width, p.camStartPixel, p.camEndPixel, height_1fr, initial_frame);*/
	data.loadMem(p.mem_bg_ptr, startposition, width, p.camStartPixel, p.camEndPixel, /*height_1fr*/1, initial_frame);

	//float2 *middle_frame = new float2[height_1fr * width]();
	float2 *middle_frame = new float2[height_1fr * used_width]();
	startposition = (height_1fr * width) * (p.numBgFrame + p.dispFrame - 1);
	data.loadMem(p.mem_data_ptr, startposition, width, p.camStartPixel, p.camEndPixel, height_1fr, middle_frame);

	/*cout << __FUNCTION__ << " startp: " << (height_1fr * width * 2 * (p.numBgFrame + p.dispFrame-1))
		<< " height: " << height_1fr << " length" << height_1fr*width
		<< " p.numBgFrame " << p.numBgFrame << " p.dispFrame " << p.dispFrame << endl;*/

	cout << "	- Initialized dispesion parameters" << endl;

	//float *column_mean = new float[width];
	float *column_mean = new float[used_width];
	float column_mean_max = 0;
	/*columnMean(height_1fr, used_width, initial_frame, column_mean, column_mean_max);*/
	for (size_t i = 0; i < used_width; i++)
	{
		column_mean[i] = initial_frame[i].x;
		if (column_mean[i] > column_mean_max)
		{
			column_mean_max = column_mean[i];
		}
	}
	delete[] initial_frame;

	float *d_column_mean;
	float2 *d_middle_frame;
	float2 *d_IFT_data;
	//std::cout<<"middle_frame:	"<<sizeof(middle_frame)<<std::endl;
	//gpuErrchk( cudaMalloc((void **)&d_column_mean, (width * sizeof(float))) );
	gpuErrchk(cudaMalloc((void **)&d_column_mean, (used_width * sizeof(float))));
	//gpuErrchk( cudaMalloc((void **)&d_middle_frame, (height_1fr * width * sizeof(float2))) );
	gpuErrchk(cudaMalloc((void **)&d_middle_frame, (height_1fr * used_width * sizeof(float2))));
	//gpuErrchk( cudaMalloc((void **)&d_IFT_data, (height_1fr * width * sizeof(float2))) );
	gpuErrchk(cudaMalloc((void **)&d_IFT_data, (height_1fr * used_width * sizeof(float2))));
	//gpuErrchk( cudaMemcpy(d_middle_frame, middle_frame, (height_1fr *  width * sizeof(float2)), cudaMemcpyHostToDevice) );
	gpuErrchk(cudaMemcpy(d_middle_frame, middle_frame, (height_1fr *  used_width * sizeof(float2)), cudaMemcpyHostToDevice));
	//gpuErrchk( cudaMemcpy(d_column_mean, column_mean, (width * sizeof(float)), cudaMemcpyHostToDevice) );
	gpuErrchk(cudaMemcpy(d_column_mean, column_mean, (used_width * sizeof(float)), cudaMemcpyHostToDevice));
	//gpuErrchk( cudaMemset(d_IFT_data, 0, (height_1fr * width * sizeof(float2))) );
	gpuErrchk(cudaMemset(d_IFT_data, 0, (height_1fr * used_width * sizeof(float2))));
	delete[] middle_frame;
	delete[] column_mean;

	//subtract<<<dimLine_w,dimLine_B>>>(height_1fr, width, d_middle_frame, d_column_mean, d_IFT_data); gpuErrchk( cudaPeekAtLastError() );
	//subtract<<<dimLine_w,dimLine_B>>>(height_1fr, used_width, d_middle_frame, d_column_mean, d_IFT_data); gpuErrchk( cudaPeekAtLastError() );
	subtract << <dimLine_uw, dimLine_B >> > (height_1fr, used_width, d_middle_frame, d_column_mean, d_IFT_data); gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaFree(d_column_mean));
	gpuErrchk(cudaFree(d_middle_frame));

	float2 *d_fringe_zp;
	gpuErrchk(cudaMalloc((void **)&d_fringe_zp, (height_1fr * width_2x * sizeof(float2))));
	gpuErrchk(cudaMemset(d_fringe_zp, 0, (height_1fr * width_2x * sizeof(float2))));  //memset for zeropad

	//FFT(height_1fr, width, d_IFT_data, d_IFT_data);
	FFT(height_1fr, used_width, d_IFT_data, d_IFT_data);

	//zero_pad<<<dimGrid_w,dimGrid_B>>>(height_1fr, width, d_IFT_data, d_fringe_zp); gpuErrchk( cudaPeekAtLastError() );
	//zero_pad<<<dimGrid_w,dimGrid_B>>>(height_1fr, width, used_width, d_IFT_data, d_fringe_zp); gpuErrchk( cudaPeekAtLastError() );
	zero_pad << <dimGrid_uw, dimGrid_B >> > (height_1fr, width, used_width, d_IFT_data, d_fringe_zp); gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaFree(d_IFT_data));

	IFT(height_1fr, width_2x, dimGrid_w2, dimGrid_B, d_fringe_zp, d_fringe_zp);

	gpuErrchk(cudaMalloc((void **)&d_fringe_interp, (height_1fr * width_2x * sizeof(float2))));
	gpuErrchk(cudaMemset(d_fringe_interp, 0, (height_1fr * width_2x * sizeof(float2))));

	interp.procInterp(height_1fr, width_2x, dimLine_w2, dimLine_B, d_fringe_zp, d_fringe_interp);
	gpuErrchk(cudaFree(d_fringe_zp));

	IFT(height_1fr, width_2x, dimGrid_w2, dimGrid_B, d_fringe_interp, d_fringe_interp);

	float2 *d_img_trim;
	gpuErrchk(cudaMalloc((void **)&d_img_trim, (height_1fr * width_h * sizeof(float2))));
	trim_width << <dimGrid_wh, dimGrid_B >> > (height_1fr, width_2x, 1, width_h, d_fringe_interp, d_img_trim); gpuErrchk(cudaPeekAtLastError());

	float *d_img_mag;
	gpuErrchk(cudaMalloc((void **)&d_img_mag, (height_1fr * width_h * sizeof(float))));
	magnitude_db << <dimGrid_wh, dimGrid_B >> > (height_1fr, width_h, d_img_trim, d_img_mag); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_img_trim));

	float *d_img_transpose;
	gpuErrchk(cudaMalloc((void **)&d_img_transpose, (width_h * height_1fr * sizeof(float))));
	transpose << <dimGrid_wh, dimGrid_B >> > (height_1fr, width_h, d_img_mag, d_img_transpose); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_img_mag));

	float *TD = new float[width_h * height_1fr];
	gpuErrchk(cudaMemcpy(TD, d_img_transpose, (width_h * height_1fr * sizeof(float)), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_img_transpose));

	ROIbounds.resize(4);

	//Image selectROI(height_1fr, width_h, TD);
	//selectROI.getPts();

	//ROIbounds[0] = selectROI.ROIpts[0].x+1;  
	//ROIbounds[1] = selectROI.ROIpts[0].y+1;		// Add plus one to these since rest of code was written
	//ROIbounds[2] = selectROI.ROIpts[1].x+1;		// with matlab test values, which are 1 indexed.
	//ROIbounds[3] = selectROI.ROIpts[1].y+1;		// Doesn't matter too much though.

	ROIbounds[0] = 0 + 1;		// This is for gridsearch testing, if it's necessary to compare
	ROIbounds[1] = 0 + 200;		// with matlab the same values can be entered into matlab as
	ROIbounds[2] = height_1fr;		// well (might need +/- 1). In general though, using manual
	ROIbounds[3] = width_h / 4 * 3;		// mode and writing in a2/a3 in MATLAB is easier.

	delete[] TD;

	width_ROI = ROIbounds[3] - ROIbounds[1] + 1; //122
	height_ROI = ROIbounds[2] - ROIbounds[0] + 1; //184

	if ((width_ROI < 0) || (height_ROI < 0))
	{
		throw std::invalid_argument("ROI Error");
	}

	dimGrid_wf = dim3((width_2x - 1) / TILE_WIDTH + 1, (height_ROI - 1) / TILE_WIDTH + 1, 1);
	dimGrid_wROI = dim3((width_ROI - 1) / TILE_WIDTH + 1, (height_ROI - 1) / TILE_WIDTH + 1, 1);

	gpuErrchk(cudaMalloc((void **)&d_fringe_frame, (height_ROI * width_2x * sizeof(float2))));
	trim_height << <dimGrid_w2, dimGrid_B >> > (height_1fr, width_2x, ROIbounds[0], ROIbounds[2], d_fringe_interp, d_fringe_frame); gpuErrchk(cudaPeekAtLastError());

	//cudaFree(d_fringe_interp); // Keep these in memory for now
	//cudaFree(d_fringe_frame);  // as they are used later.
}

void Dispersion::gridsearch()
{
	w.resize(width_2x);
	for (int i = 0; i < width_2x; i++)
		w[i] = i - width + 1;

	FFT(height_ROI, width_2x, d_fringe_frame, d_fringe_frame);

	fphi = new float2[width_2x];

	// prep a fft plan for getParam call inside secondOrder() and thirdOrder().
	int w2ROIh[2] = { width_2x, height_ROI };
	cufftPlanMany(&plan_ROIw, 1, w2ROIh, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, height_ROI);
	//log the grid searched a2 and a3 @ TaoXu
	/*std::ofstream logfile("diperstion_optimization.txt",std::ofstream::app);
	logfile<<"grid search:"<<std::endl;
	logfile.close();*/

	/*secondOrder();
	thirdOrder();*/
	// twoDOrderSearch();
	recursiveGridSearch();
	cufftDestroy(plan_ROIw);

	genFphi();

	gpuErrchk(cudaFree(d_fringe_frame));
	delete[] fphi;

	cout << "	- Calculated gridsearch parameters" << endl;
}

void Dispersion::secondOrder()
{
	float temp = 0;
	a2.resize(p.dispTests);
	vector<float> R_a2(p.dispTests, 0);
	linSpace(-0.6f, 0.6f, p.dispTests, a2);

	for (int i = 0; i < p.dispTests; i++)
	{
		for (int j = 0; j < width_2x; j++)
		{
			temp = (-a2[i] * powf(w[j], 2)) / width_2x;
			fphi[j].x = cos(temp);
			fphi[j].y = sin(temp);
		}
		gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));
		getParam(i, R_a2); // d_fphi is used in here
	}

	// Get the index of the min value.
	int na2 = 0;
	std::ofstream logfile("a2.csv", std::ofstream::out);
	for (int i = 1; i < p.dispTests; i++)
	{
		logfile << R_a2[i] << std::endl;
		if (R_a2[i] < R_a2[na2])
			na2 = i;
	}
	std::cout << " a2 idx: " << na2 << std::endl;
	std::cout << " a2 : " << a2[na2] << std::endl;
	dispParam[0] = a2[na2];
	logfile.close();
}

void Dispersion::thirdOrder()
{
	float temp = 0;
	vector<float> a3(p.dispTests);
	linSpace(-0.1f, 0.1f, p.dispTests, a3);
	for (int i = 0; i < p.dispTests; i++)
		a3[i] /= width_2x;
	vector<float> R_a3(p.dispTests, 0);

	for (int i = 0; i < p.dispTests; i++)
	{
		for (int j = 0; j < width_2x; j++)
		{
			//temp =  dispParam[0] *  w[j] * w[j] + (a3[i] * w[j] * w[j] * w[j]);
			temp = (dispParam[0] * powf(w[j], 2)) + (a3[i] * powf(w[j], 3));
			temp = -temp / width_2x;
			fphi[j].x = cos(temp);
			fphi[j].y = sin(temp);
		}
		gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));
		getParam(i, R_a3); // d_fphi is used in here
	}

	// Get the index of the min value.
	int na3 = 0;
	std::ofstream logfile("a3.csv", std::ofstream::out);
	for (int i = 1; i < p.dispTests; i++)
	{
		logfile << R_a3[i] << std::endl;
		if (R_a3[i] < R_a3[na3])
			na3 = i;
	}
	std::cout << " a3 idx: " << na3 << std::endl;
	std::cout << " a3 : " << a3[na3] << std::endl;
	dispParam[1] = a3[na3];
	dispParam[2] = dispParam[3] = 0;
	logfile.close();
}

void Dispersion::twoDOrderSearch()
{
	// init second order
	float temp = 0;
	a2.resize(p.dispTests);
	linSpace(-0.6f, 0.6f, p.dispTests, a2);

	// init third order
	vector<float> a3(p.dispTests);
	linSpace(-0.1f, 0.1f, p.dispTests, a3);
	for (int i = 0; i < p.dispTests; i++)
		a3[i] /= width_2x;

	std::vector<float> R_a23(p.dispTests*p.dispTests, 0);

	for (int i = 0; i < p.dispTests; i++)
	{
		for (size_t j = 0; j < p.dispTests; j++)
		{
			for (int k = 0; k < width_2x; k++)
			{
				//temp =  dispParam[0] *  w[j] * w[j] + (a3[i] * w[j] * w[j] * w[j]);
				temp = (a2[i] * powf(w[k], 2)) + (a3[j] * powf(w[k], 3));
				temp = -temp / width_2x;
				fphi[k].x = cos(temp);
				fphi[k].y = sin(temp);
			}
			gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));
			getParam(j + i*p.dispTests, R_a23); // d_fphi is used in here
		}
	}

	int idx = 0;
	std::ofstream logfile("tewD.csv", std::ofstream::out);
	logfile << R_a23[0] << ",";
	for (int i = 1; i < p.dispTests*p.dispTests; i++)
	{
		logfile << R_a23[i];
		if ((i + 1) % p.dispTests == 0)
		{
			logfile << std::endl;
		}
		else
		{
			logfile << ",";
		}
		if (R_a23[i] < R_a23[idx])
			idx = i;
	}
	std::cout << "idx 2: " << int(idx / p.dispTests) << " - idx 3: " << int(idx % p.dispTests) << std::endl;
	std::cout << "a2: " << a2[int(idx / p.dispTests)] << " - a3: " << a3[int(idx % p.dispTests)] << std::endl;
	dispParam[0] = a2[int(idx / p.dispTests)];
	dispParam[1] = a3[int(idx % p.dispTests)];
	dispParam[2] = dispParam[3] = 0;
	logfile.close();
}

void Dispersion::recursiveGridSearch()
{
	int cubeDispTest = 10;
	std::vector<float> R_a23(cubeDispTest * cubeDispTest, 0);
	float temp = 0;
	// init step
	vector<float> a3(cubeDispTest);
	float a2s=-0.6f, a2e=0.6f, a3s=-0.1f, a3e=0.1f;
	linSpace(a3s, a3e, cubeDispTest, a3);
	for (int i = 0; i < cubeDispTest; i++)
		a3[i] /= width_2x;
	a2.resize(cubeDispTest);
	linSpace(a2s, a2e, cubeDispTest, a2);
	// recursive grid search
	float minRa23 = -10000000000;
	int foundX = -1, foundY = -1;
	/*std::ofstream logfile("recursiveGridSearch.csv", std::ofstream::out);
	std::ofstream log23("a23.csv", std::ofstream::out);*/
	while (true)
	{
		//log recursive a2 and a3
		/*for (size_t i = 0; i < cubeDispTest; i++)
		{
			logfile << a3[i] << ",";
		}
		logfile << std::endl;
		for (size_t i = 0; i < cubeDispTest; i++)
		{
			logfile << a2[i] << ",";
		}
		logfile << std::endl;*/

		for (int i = 0; i < cubeDispTest; i++)
		{
			for (size_t j = 0; j < cubeDispTest; j++)
			{
				for (int k = 0; k < width_2x; k++)
				{
					//temp =  dispParam[0] *  w[j] * w[j] + (a3[i] * w[j] * w[j] * w[j]);
					temp = (a2[i] * powf(w[k], 2)) + (a3[j] * powf(w[k], 3));
					temp = -temp / width_2x;
					fphi[k].x = cos(temp);
					fphi[k].y = sin(temp);
				}
				gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));
				getParam(j + i*cubeDispTest, R_a23); // d_fphi is used in here
				/*if (minRa23 < R_a23[j + i*cubeDispTest])
				{
					foundX = j;
					foundY = i;
					minRa23 = R_a23[j + i*cubeDispTest];
				}*/
			}
		}
		int idx = 0;
		/*logfile << R_a23[0] << ",";*/
		for (int i = 1; i < cubeDispTest*cubeDispTest; i++)
		{
			/*logfile << R_a23[i];
			if ((i + 1) % cubeDispTest == 0)
			{
				logfile << std::endl;
			}
			else
			{
				logfile << ",";
			}*/
			if (R_a23[i] < R_a23[idx])
				idx = i;
		}
		foundX = idx % cubeDispTest;
		foundY = idx / cubeDispTest;
		dispParam[0] = a2[foundY];
		dispParam[1] = a3[foundX];
		dispParam[2] = dispParam[3] = 0;

		// log23 << dispParam[0] << "," << dispParam[1] << "," << R_a23[foundX + foundY*cubeDispTest] << std::endl;
		// reset to start next grid search
		if (foundX <= 0)
		{
			a3s = a3[foundX];
			a3e = a3[foundX + 1];
		}
		else if(foundX >= cubeDispTest-1)
		{
			a3s = a3[foundX - 1];
			a3e = a3[foundX];
		}
		else
		{
			a3s = a3[foundX - 1];
			a3e = a3[foundX + 1];
		}

		if (foundY <= 0)
		{
			a2s = a2[foundY];
			a2e = a2[foundY + 1];
		}
		else if (foundY >= cubeDispTest-1)
		{
			a2s = a2[foundY - 1];
			a2e = a2[foundY];
		}
		else
		{
			a2s = a2[foundY - 1];
			a2e = a2[foundY + 1];
		}
		float spacex = (a3e - a3s) / (cubeDispTest - 1), spacey = (a2e - a2s) / (cubeDispTest - 1);
		if (spacex < 5*10e-5 && spacey < 5*10e-5)
		{
			break;
		}
		linSpace(a3s, a3e, cubeDispTest, a3);
		for (int i = 0; i < cubeDispTest; i++)
			a3[i] /= width_2x;
		linSpace(a2s, a2e, cubeDispTest, a2);
		foundX = foundY = -1;
		minRa23 = -100000000;

	}
	std::cout << "a2: " << dispParam[0] << " - a3: " << dispParam[1] << std::endl;
	/*logfile.close();
	log23.close();*/
}

void Dispersion::genFphi()
{
	vector<float> phi(width_2x, 0.0);

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < width_2x; j++)
			//phi[j] = phi[j] + dispParam[i] * powf((j+1.f), (i+2));
			phi[j] = phi[j] + dispParam[i] * powf((j - width + 1.f), (i + 2));

	// debug for the manual dispersion @TaoXu
	/*std::ofstream logfile("mannul_dispersion.txt",std::ofstream::app);
	for(size_t i = 0; i < width_2x; ++i)
	{
		logfile<<phi[i]<<"	";
	}
	logfile<<endl;
	logfile.close();*/

	// need to fix @brian
	// j - width + 1;

	float temp = 0;
	for (int i = 0; i < width_2x; i++)
	{
		temp = -phi[i] / width_2x;
		fphi[i].x = cos(temp);
		fphi[i].y = sin(temp);
	}
	gpuErrchk(cudaMemcpy(d_fphi, fphi, (width_2x * sizeof(float2)), cudaMemcpyHostToDevice));

	repmat << <dimLine_w2, dimLine_B >> > (p.batchAscans, width_2x, d_fphi); gpuErrchk(cudaPeekAtLastError());
}

void Dispersion::previewComp()
{

	FFT(height_1fr, width_2x, d_fringe_interp, d_fringe_interp);

	float2 *d_comp;
	gpuErrchk(cudaMalloc((void **)&d_comp, (height_1fr * width_2x * sizeof(float2))));

	// matlab scales both real and imag compenets of fhi with the real component of S
	phi_multiply << <dimLine_w2, dimLine_B >> > (height_1fr, width_2x, d_fphi, d_fringe_interp, d_comp); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_fringe_interp));

	IFT(height_1fr, width_2x, dimGrid_w2, dimGrid_B, d_comp, d_comp);

	float2 *d_comp_trim;
	gpuErrchk(cudaMalloc((void **)&d_comp_trim, (height_1fr * width_h * sizeof(float2))));
	trim_width << <dimGrid_w2, dimGrid_B >> > (height_1fr, width_2x, 1, width_h, d_comp, d_comp_trim); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_comp));

	float *d_comp_mag;
	gpuErrchk(cudaMalloc((void **)&d_comp_mag, (height_1fr * width_h * sizeof(float))));
	magnitude_db << <dimGrid_wh, dimGrid_B >> > (height_1fr, width_h, d_comp_trim, d_comp_mag); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_comp_trim));

	float *d_comp_transpose;
	gpuErrchk(cudaMalloc((void **)&d_comp_transpose, (width_h * height_1fr * sizeof(float))));
	transpose << <dimGrid_wh, dimGrid_B >> > (height_1fr, width_h, d_comp_mag, d_comp_transpose); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_comp_mag));

	float *TD_comp_preview = new float[width_h * height_1fr];
	gpuErrchk(cudaMemcpy(TD_comp_preview, d_comp_transpose, (width_h * height_1fr * sizeof(float)), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_comp_transpose));

	Image comp_image(height_1fr, width_h, TD_comp_preview);
	comp_image.dspl();

	delete[] TD_comp_preview;
	cout << "	- Generated compensated preview image" << endl;
}

void Dispersion::getParam(int it, vector<float> &result_vector)
{
	// matlab scales both real and imag compenets of fphi with the real component of S.
	float2 *d_S;
	gpuErrchk(cudaMalloc((void **)&d_S, (height_ROI * width_2x * sizeof(float2))));
	phi_multiply << <dimLine_w2, dimLine_B >> > (height_ROI, width_2x, d_fphi, d_fringe_frame, d_S); gpuErrchk(cudaPeekAtLastError());

	cufftErrchk(cufftExecC2C(plan_ROIw, d_S, d_S, CUFFT_INVERSE));
	scale_IFT << <dimGrid_w2, dimGrid_B >> > (height_ROI, width_2x, w2Recip, d_S);

	float2 *d_S_ROI;
	gpuErrchk(cudaMalloc((void **)&d_S_ROI, (height_ROI * width_ROI * sizeof(float2))));
	trim_width << <dimGrid_wf, dimGrid_B >> > (height_ROI, width_2x, ROIbounds[1], ROIbounds[3], d_S, d_S_ROI); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_S));

	float *d_S_abs;
	gpuErrchk(cudaMalloc((void **)&d_S_abs, (height_ROI * width_ROI * sizeof(float))));
	magnitude << <dimGrid_wROI, dimGrid_B >> > (height_ROI, width_ROI, d_S_ROI, d_S_abs); gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaFree(d_S_ROI));

	float s = 0;
	sumElements(height_ROI, width_ROI, d_S_abs, s);
	divide << <dimGrid_wROI, dimGrid_B >> > (height_ROI, width_ROI, d_S_abs, s, d_S_abs); gpuErrchk(cudaPeekAtLastError());

	float *d_S_log;
	gpuErrchk(cudaMalloc((void **)&d_S_log, (height_ROI * width_ROI * sizeof(float))));
	d_log << <dimGrid_wROI, dimGrid_B >> > (height_ROI, width_ROI, d_S_abs, d_S_log); gpuErrchk(cudaPeekAtLastError());

	multiply << <dimGrid_wROI, dimGrid_B >> > (height_ROI, width_ROI, d_S_abs, d_S_log, d_S_abs); gpuErrchk(cudaPeekAtLastError());
	sumElements(height_ROI, width_ROI, d_S_abs, result_vector[it]);
	gpuErrchk(cudaFree(d_S_abs));

	result_vector[it] = -result_vector[it];

	/*********************************************************************************************************************/
																											  /* Preview */
	gpuErrchk(cudaFree(d_S_log));
	//float *d_S_transpose;
	//gpuErrchk(cudaMalloc((void **)&d_S_transpose, (width_ROI * height_ROI * sizeof(float))));
	//transpose << <dimGrid_wROI, dimGrid_B >> > (height_ROI, width_ROI, d_S_log, d_S_transpose); gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaFree(d_S_log));

	//float *compPrev = new float[width_ROI * height_ROI];
	//gpuErrchk(cudaMemcpy(compPrev, d_S_transpose, (width_ROI * height_ROI * sizeof(float)), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaFree(d_S_transpose));

	////// DOM
	///*cv::Mat curImg = cv::Mat(height_ROI, width_ROI, CV_32FC1, compPrev);
	//Image::normImg(curImg);
	//cv::normalize(curImg, curImg, 255.0, 0.0, cv::NORM_MINMAX);
	//cv::Mat newImg;
	//curImg.convertTo(newImg, CV_8UC1);
	//result_vector[it] = this->m_dom.get_sharpness(curImg, 3);*/

	///*Image dispCompImg(height_ROI, width_ROI, compPrev);
	//dispCompImg.dsplGS();*/

	//delete[] compPrev;
}

void Dispersion::linSpace(float min, float max, int pts, vector<float> &result_vector)
{
	float space = (max - min) / (pts - 1);
	result_vector[0] = min;

	for (int i = 1; i < pts; i++)
		result_vector[i] = result_vector[i - 1] + space;
}

void Dispersion::sumElements(int height, int width, float *initial_array, float &sum)
{
	float *d_result_array;
	gpuErrchk(cudaMalloc((void **)&d_result_array, (height * width * sizeof(float))));
	gpuErrchk(cudaMemcpy(d_result_array, initial_array, (height * width * sizeof(float)), cudaMemcpyDeviceToDevice));

	float *d_temp_sum;
	gpuErrchk(cudaMalloc((void **)&d_temp_sum, (height * width * sizeof(float))));

	int sharedMemSize = TILE_WIDTH * TILE_WIDTH * sizeof(float);
	float sumHeight = height;
	float sumWidth = width;
	dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	int k = (height * width);

	while (k != 0)
	{
		d_sum_elements << <dimGrid, dimBlock, sharedMemSize >> > (sumHeight, sumWidth, d_result_array, d_temp_sum); gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaMemcpy(d_result_array, d_temp_sum, (sumHeight * sumWidth * sizeof(float)), cudaMemcpyDeviceToDevice));

		sumHeight = ceil(sumHeight / TILE_WIDTH);
		sumWidth = ceil(sumWidth / TILE_WIDTH);;
		dimGrid.x = (sumWidth - 1) / TILE_WIDTH + 1;
		dimGrid.y = (sumHeight - 1) / TILE_WIDTH + 1;
		k /= TILE_WIDTH*TILE_WIDTH;
	}

	gpuErrchk(cudaMemcpy(&sum, &d_result_array[0], sizeof(float), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_temp_sum));
	gpuErrchk(cudaFree(d_result_array));
}