#include "output.cuh"
#include "TiffWriter.h"
#include "utilities.h"
#include<windows.h>
//#include "cuda_profiler_api.h"
#include <iostream>
#include <list>
#include <omp.h>
#include <fstream>
#include <ctime>

using namespace std;
//using namespace cv;

void Output::CorrectDiff(cv::Mat &srcImage)
{
	//Magick::Image magick_image;
	//magick_image.read(path + "[" + std::to_string(static_cast<long long>(0)) + "]");
	//Magick::Blob blob;
	//magick_image.write(&blob);
	//cv::Mat srcImage(magick_image.rows(), magick_image.columns(), CV_8UC1);
	////cv::Mat(magick_image.rows(), magick_image.columns(), (void *)blob.data());
	////image.write(0, 0, w, h, "BGR", Magick::CharPixel, opencvImage.data);
	//std::memcpy(srcImage.data, blob.data(), magick_image.rows()*magick_image.columns());
	//cv::Mat srcImage = cv::imread(path);


	vector<cv::Mat> channels;
	cv::split(srcImage.clone(), channels);
	cv::Mat gray = channels.at(0);
	int height = gray.size().height;
	int width = gray.size().width;
	vector<int> diff(width, 0);
	cv::Point_<uchar> *p;

	for (int i = height / 3; i < height; i++) {
		for (int j = 0; j < width; j++) {
			p = gray.ptr<cv::Point_<uchar> >(i, j);
			p->x = 0;
		}

	}
	cv::threshold(gray.clone(), gray, 50, 255, cv::THRESH_BINARY);
	Sobel(gray.clone(), gray, CV_8U, 0, 1, 7);

	cv::Mat element1 = cv::getStructuringElement(0, cv::Size(1, 3), cv::Point(0, 0));

	erode(gray.clone(), gray, element1);


	cv::Mat temp = cv::Mat::zeros(height, width, CV_8U) + 255;
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	sort(contours.begin(), contours.end(),
		[](vector<cv::Point> a, vector<cv::Point> b) {return a.size() > b.size(); });


	int count = contours.size();

	int max_ = 5;
	for (int i = 0; i < count - max_; i++) {
		contours.erase(contours.begin() + max_);
	}
	drawContours(temp, contours, -1, (0, 0, 0), 1);

	cv::Point_<ushort> *p1;
	cv::Mat y(1, width, CV_16U, cv::Scalar::all(0));
	for (int j = 0; j < width; j++) {
		int flag = 1;
		p1 = y.ptr<cv::Point_<ushort> >(0, j);

		for (int i = 0; i < height; i++) {
			p = temp.ptr<cv::Point_<uchar> >(i, j);
			if (p->x == 0) {
				flag = 0;
				p1->x = i;
				break;
			}
		}
		if (flag == 1 && j != 0) {

			p = y.ptr<cv::Point_<uchar> >(0, j - 1);
			p1->x = p->x;
		}

	}

	int mid = cv::mean(y)[0];
	for (int j = 0; j < width; j++) {
		p1 = y.ptr<cv::Point_<ushort> >(0, j);


		if (j != 0) diff[j] = (int)(mid - p1->x + 0.5);//+ diff[j];
	}
	this->blurDiff(diff);
	ofstream outFile;
	outFile.open("../conf/diff.csv", ios::out);
	for (int j = 0; j < width; j++)
	{
		outFile << diff[j] << endl;
	}
	for (size_t i = width; i < 4096; ++i)
	{
		outFile << 0 << endl;
	}
	outFile.close();
	outFile.clear();
}

void Output::blurDiff(vector<int> &diff)
{
	float sum = 0;
	int size = 100;
	int count = 0;
	for (int i = 0; i < diff.size(); i++)
	{
		count = 0;
		sum = 0;
		for (int j = -size; j <= size; j++)
		{
			if (i + j >= 0 && i + j < diff.size() && diff[i] != 0)
			{
				count++;
				sum += diff[i + j];
			}
		}

		if (count > 0)
		{

			diff[i] = (int)sum / count;
		}
	}
}

void Output::edgeCorrect(float *aloneAscan, int diff, int width)
{
	if (diff > 0) {
		for (int i = width - 1; i >= 0; i--) {
			if (i - diff >= 0) {
				aloneAscan[i] = aloneAscan[i - diff];
			}
			else {
				aloneAscan[i] = 0;
			}
		}
	}
	else {
		for (int i = 0; i <= width - 1; i++) {
			if (i - diff < width) {
				aloneAscan[i] = aloneAscan[i - diff];
			}
			else {
				aloneAscan[i] = 0;
			}
		}
	}
}

void Output::flatOut(cv::Mat &srcImage) {
	cv::Point_<uchar> *p1;
	cv::Point_<uchar> *p2;

	for (int j = 0; j<srcImage.size().width; j++) {
		if (diffs[j]>0) {
			for (int i = srcImage.size().height - 1; i >= 0; i--) {
				p1 = srcImage.ptr<cv::Point_<uchar> >(i, j);
				if (i - diffs[j]>0) {
					p2 = srcImage.ptr<cv::Point_<uchar> >(i - diffs[j], j);
					p1->x = p2->x;
				}
				else {
					p1->x = 0;
				}
			}
		}

		else {
			for (int i = 0; i <= srcImage.size().height - 1; i++) {
				p1 = srcImage.ptr<cv::Point_<uchar> >(i, j);
				if (i - diffs[j]<srcImage.size().height) {
					p2 = srcImage.ptr<cv::Point_<uchar> >(i - diffs[j], j);
					p1->x = p2->x;
				}
				else {
					p1->x = 0;
				}
			}
		}
	}
}

//static function for shift and average image
void Output::shiftAndAvg(std::vector<cv::Mat>& mat_vec)
{
	if (mat_vec.size() == 1)
	{
		return;
	}
	cv::Mat src = mat_vec[0];
	unsigned int count = 1;
	//while (begin_iter != mat_vec.end())
	omp_set_dynamic(0); //disable dynamic thread team size
	omp_set_num_threads(mat_vec.size() - 1); // fix the number of threads to 2
	//while (begin_iter != mat_vec.end())
#pragma omp parallel
	{
		for (size_t i = 1; i < mat_vec.size(); ++i)
		{
			unsigned int threadID = omp_get_thread_num();
			if (threadID == i - 1)
			{
				cv::Point2d cr = cv::phaseCorrelate(cv::Mat_<float>(src), cv::Mat_<float>(mat_vec[i]));
				// if the shifted pixel large than 20, than do not do any thing
				if (abs(cr.x) >= 20 || abs(cr.y) >= 20)
				{
					continue;
				}
				cv::Mat trans_mat = (cv::Mat_<float>(2, 3) << 1, 0, cr.x, 0, 1, cr.y);
				cv::Mat dst;
				cv::warpAffine(mat_vec[i], dst, trans_mat, mat_vec[0].size());
#pragma omp critical
				mat_vec[0] += dst;

#pragma omp atomic
				count += 1;
			}
#pragma omp barrier
		}
	}
	mat_vec[0] /= count;
}


Output::Output(Parameters &p, Data &da, Interpolation &i, Dispersion &di, Background &b) : Helper(p), data(da), interp(i), disp(di), bg(b)
{
	cout << "Data processing" << endl;

	height_ba = p.numAscansPerBscan / p.batchAscans;
	std::cout << "height_ba:" << height_ba << std::endl;
	width_ba = p.numCameraPixels*p.batchAscans;
	width_2xba = (p.numCameraPixels * 2)*p.batchAscans;
	//frames_tot = p.numBScans*p.batchFrames;
	frames_tot = p.numBScans*p.batchFrames*p.avgBscan;
	used_width_ba = used_width*p.batchAscans;
	used_width_2xba = (used_width * 2)*p.batchAscans;

	dimGrid_w = dim3((width - 1) / TILE_WIDTH + 1, (height_bfr - 1) / TILE_WIDTH + 1, 1);
	dimGrid_w2 = dim3((width_2x - 1) / TILE_WIDTH + 1, (height_bfr - 1) / TILE_WIDTH + 1, 1);
	dimGrid_wt = dim3((width_trm - 1) / TILE_WIDTH + 1, (height_bfr - 1) / TILE_WIDTH + 1, 1);

	dimGrid_uw = dim3((used_width - 1) / TILE_WIDTH + 1, (height_bfr - 1) / TILE_WIDTH + 1, 1);

	dimLine_wba = dim3((width_ba + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
	dimLine_w2xba = dim3((width_2xba + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
	// dimLine_w2xba = dim3((width_2xba+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, height_ba, 1);
	dimLine_wtba = dim3(((width_trm*p.batchAscans) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);

	dimLine_uwba = dim3((used_width_ba + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);

	//define tiffhandler
	this->pageSize.width = width_trm;
	this->pageSize.height = height_1fr;

	//load diff
	this->diffs = std::vector<int>(p.numAscansPerBscan, 0);
	FILE *fp = fopen("../conf/diff.csv", "r");
	float item;
	int count = 0;
	while (count < p.numAscansPerBscan) {
		fscanf(fp, "%f", &item);
		// diff.push_back(mean-int(item+0.5));
		this->diffs[count++] = item;
		if (feof(fp)) break;
	}
	fclose(fp);
	//for (int i = 0; i < diffs.size(); i++) {
	//	cout << diffs[i] << endl;
	//}

	this->process_type = "";
}

Output::~Output()
{
	//this->tiffhandler.CloseTiffFile();
	this->diffs.swap(std::vector<int>());
}

void Output::process(std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData)
{
	//std::clock_t start = clock();
	//cudaMemGetInfo(&mem_avail, &mem_total);
	//std::cout << "Debug: " << __FUNCTION__ << "0 mem avail: " << mem_avail << " total: " << mem_total << std::endl;
	this->process_type = datatype;
	// Hold the full processed image stack in memory.
	float *processed_data_array = new float[width_trm * height_bfr * frames];
	//std::cout << "	width_trm:" << width_trm << "	height_bfr:" << height_bfr << "	frames:" << frames << std::endl;
	omp_set_dynamic(0); //disable dynamic thread team size
	omp_set_num_threads(3); // fix the number of threads to 2
	/*std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for openmp preparation :	" << clock() - start << "ms" << std::endl;
	start = clock();*/
#pragma omp parallel
	{
		initResources();
#pragma omp barrier
		//std::cout<<"Debug: "<<__FUNCTION__<<" process 1"<<std::endl;
		//gpuErrchk( cudaProfilerStart() );
		for (int i = 0; i < (frames + 1) / 2; i++)
		{
			processData(i, processed_data_array, previewFFTrawData);
		}
	}
	/*std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for openmp process :	" << clock() - start << "ms" << std::endl;

	start = clock();*/
	gpuErrchk(cudaMemcpy(processed_data_array, largeBuff, (frames*width_trm * height_bfr * sizeof(float)), cudaMemcpyDeviceToHost));
	/*std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for openmp DeviceToHost :	" << clock() - start << "ms" << std::endl;

	start = clock();*/
	freeResources();
	/*std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for free GPU source :	" << clock() - start << "ms" << std::endl;

	start = clock();*/
	writeToDisk(processed_data_array, datatype, bmpRawData);
	//std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for write to disk :	" << clock() - start << "ms" << std::endl;
	delete[] processed_data_array;
}

void Output::initResources()
{
#pragma omp master
	{
		gpuErrchk(cudaMallocHost((void**)&h_buff_1, height_bfr*used_width * sizeof(int16_t)));
		gpuErrchk(cudaMallocHost((void**)&h_buff_2, height_bfr*used_width * sizeof(int16_t)));
		gpuErrchk(cudaMallocHost((void**)&h_buff_3, height_bfr*used_width * sizeof(int16_t)));
		gpuErrchk(cudaMallocHost((void**)&h_buff_4, height_bfr*used_width * sizeof(int16_t)));


		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaStreamCreate(&stream1));
		gpuErrchk(cudaStreamCreate(&stream2));

		//int m[2] = {width, height_bfr};
		int m[2] = { used_width, height_bfr };
		cufftErrchk(cufftPlanMany(&plan_w, 1, m, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, height_bfr));
		cufftErrchk(cufftSetStream(plan_w, stream1));
		cufftErrchk(cufftPlanMany(&plan_w_s2, 1, m, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, height_bfr));
		cufftErrchk(cufftSetStream(plan_w_s2, stream2));

		int n[2] = { width_2x, height_bfr };
		cufftErrchk(cufftPlanMany(&plan_w2, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, height_bfr));
		cufftErrchk(cufftSetStream(plan_w2, stream1));
		cufftErrchk(cufftPlanMany(&plan_w2_s2, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, height_bfr));
		cufftErrchk(cufftSetStream(plan_w2_s2, stream2));

		//gpuErrchk( cudaMalloc((void **)&d_raw_data, (height_bfr * width * sizeof(uint16_t))) );
		//gpuErrchk( cudaMalloc((void **)&d_raw_data, (height_bfr * used_width * sizeof(uint16_t))) );

		// Joe : June 6 : Attempting to double the size of d_raw_data to allow buffering
		//gpuErrchk( cudaMalloc((void **)&d_raw_data, ( height_bfr * used_width * sizeof(uint16_t))) );
		gpuErrchk(cudaMalloc((void **)&d_raw_data_buf_1, (height_bfr * used_width * sizeof(int16_t))));
		gpuErrchk(cudaMalloc((void **)&d_raw_data_buf_2, (height_bfr * used_width * sizeof(int16_t))));
		gpuErrchk(cudaMalloc((void **)&d_raw_data_buf_3, (height_bfr * used_width * sizeof(int16_t))));
		gpuErrchk(cudaMalloc((void **)&d_raw_data_buf_4, (height_bfr * used_width * sizeof(int16_t))));

		//gpuErrchk( cudaMalloc((void **)&d_proc_buff_0, (height_bfr * width * sizeof(float2))) ); // used_width
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_0, (height_bfr * used_width * sizeof(float2))));
		//gpuErrchk( cudaMalloc((void **)&d_proc_buff_1, (height_bfr * width_2x * sizeof(float2))) );
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_1, (height_bfr * width_2x * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_2, (height_bfr * width_2x * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_3, (height_bfr * used_width * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_4, (height_bfr * width_2x * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_5, (height_bfr * width_2x * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_trm, (height_bfr * width_trm * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_db, (height_bfr * width_trm * sizeof(float))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_trns, (width_trm * height_bfr * sizeof(float))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_trm_1, (height_bfr * width_trm * sizeof(float2))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_db_1, (height_bfr * width_trm * sizeof(float))));
		gpuErrchk(cudaMalloc((void **)&d_proc_buff_trns_1, (width_trm * height_bfr * sizeof(float))));
		gpuErrchk(cudaMalloc((void **)&largeBuff, (((frames + 1) / 2) * 2 * width_trm * height_bfr * sizeof(float))));


		// Joe : June 6 : Loading in two frames
		if (frames >= 1)
		{
			//data.loadFile(p.fnameData, ((height_1fr * width_2x) * (p.numBgFrame + ((0 + 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_1);
			data.loadMem(p.mem_data_ptr, ((height_1fr * width) * (p.numBgFrame + ((0 + 1 - 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_1);

			//gpuErrchk( cudaMemcpy(d_raw_data_buf_1, h_buff_1, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice) );
			gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_1, h_buff_1, (height_bfr * used_width * sizeof(int16_t)), cudaMemcpyHostToDevice, stream1));
			// std::cout<<((height_1fr * width_2x) * (p.numBgFrame + ((0 + 1)*p.batchFrames)))<<std::endl;
			if (frames >= 2)
			{
				//data.loadFile(p.fnameData, ((height_1fr * width_2x) * (p.numBgFrame + ((1 + 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_3);
				data.loadMem(p.mem_data_ptr, ((height_1fr * width) * (p.numBgFrame + ((1 + 1 - 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_3);
				gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_3, h_buff_3, (height_bfr * used_width * sizeof(int16_t)), cudaMemcpyHostToDevice, stream2));
				//gpuErrchk( cudaMemcpy(d_raw_data_buf_3, h_buff_3, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice) );
				//std::cout<<((height_1fr * width_2x) * (p.numBgFrame + ((1 + 1)*p.batchFrames)))<<std::endl;
				if (frames >= 3)
				{
					//data.loadFile(p.fnameData, ((height_1fr * width_2x) * (p.numBgFrame + ((2 + 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_2);
					data.loadMem(p.mem_data_ptr, ((height_1fr * width) * (p.numBgFrame + ((2 + 1 - 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_2);
					//gpuErrchk( cudaMemcpy(d_raw_data_buf_2, h_buff_2, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice) );
					gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_2, h_buff_2, (height_bfr * used_width * sizeof(int16_t)), cudaMemcpyHostToDevice, stream1));
					//std::cout<<((height_1fr * width_2x) * (p.numBgFrame + ((2 + 1)*p.batchFrames)))<<std::endl;

					if (frames >= 4)
					{
						//data.loadFile(p.fnameData, ((height_1fr * width_2x) * (p.numBgFrame + ((3 + 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_4);
						data.loadMem(p.mem_data_ptr, ((height_1fr * width) * (p.numBgFrame + ((3 + 1 - 1)*p.batchFrames))), width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_4);
						//gpuErrchk( cudaMemcpy(d_raw_data_buf_4, h_buff_4, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice) );
						gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_4, h_buff_4, (height_bfr * used_width * sizeof(int16_t)), cudaMemcpyHostToDevice, stream2));
						//std::cout<<((height_1fr * width_2x) * (p.numBgFrame + ((3 + 1)*p.batchFrames)))<<std::endl;
					}
				}
			}
		}
		gpuErrchk(cudaDeviceSynchronize());
	}
}



void Output::processData(int it, float *proc_data_piece, std::vector<float>& previewFFTrawData)
{
	std::clock_t start = clock();
	int threadID;
	threadID = omp_get_thread_num();

	if (threadID == 0) {
		gpuErrchk(cudaMemsetAsync(d_proc_buff_0, 0, (height_bfr * used_width * sizeof(float2)), stream1));
		gpuErrchk(cudaMemsetAsync(d_proc_buff_3, 0, (height_bfr * used_width * sizeof(float2)), stream2));

		if (it % 2 == 0) {
			/*subtract << <dimLine_uwba, dimLine_B, 0, stream1 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_1, bg.d_bg, d_proc_buff_0); gpuErrchk(cudaPeekAtLastError());
			subtract << <dimLine_uwba, dimLine_B, 0, stream2 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_3, bg.d_bg, d_proc_buff_3); gpuErrchk(cudaPeekAtLastError()); */
			div_subtract << <dimLine_uwba, dimLine_B, 0, stream1 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_1, bg.d_bg, bg.d_bg_mask, d_proc_buff_0); gpuErrchk(cudaPeekAtLastError());
			div_subtract << <dimLine_uwba, dimLine_B, 0, stream2 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_3, bg.d_bg, bg.d_bg_mask, d_proc_buff_3); gpuErrchk(cudaPeekAtLastError());
		}
		else {
			/*subtract << <dimLine_uwba, dimLine_B, 0, stream1 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_2, bg.d_bg, d_proc_buff_0); gpuErrchk(cudaPeekAtLastError());
			subtract << <dimLine_uwba, dimLine_B, 0, stream2 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_4, bg.d_bg, d_proc_buff_3); gpuErrchk(cudaPeekAtLastError());*/
			div_subtract << <dimLine_uwba, dimLine_B, 0, stream1 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_2, bg.d_bg, bg.d_bg_mask, d_proc_buff_0); gpuErrchk(cudaPeekAtLastError());
			div_subtract << <dimLine_uwba, dimLine_B, 0, stream2 >> > (height_ba, used_width_ba, used_width, d_raw_data_buf_4, bg.d_bg, bg.d_bg_mask, d_proc_buff_3); gpuErrchk(cudaPeekAtLastError());
		}
		//subtract<<<dimLine_uwba,dimLine_B,0,stream1>>>(height_ba, used_width_ba, d_raw_data_buf_1, bg.d_bg, d_proc_buff_0); gpuErrchk( cudaPeekAtLastError() );

		cufftErrchk(cufftExecC2C(plan_w, d_proc_buff_0, d_proc_buff_0, CUFFT_FORWARD));
		cufftErrchk(cufftExecC2C(plan_w_s2, d_proc_buff_3, d_proc_buff_3, CUFFT_FORWARD));
		//print_gpu_data(1, width_2x, d_proc_buff_0, "d_proc_buff_1-1");

		gpuErrchk(cudaMemsetAsync(d_proc_buff_1, 0, (height_bfr * width_2x * sizeof(float2)), stream1));
		gpuErrchk(cudaMemsetAsync(d_proc_buff_4, 0, (height_bfr * width_2x * sizeof(float2)), stream2));

		zero_pad << <dimGrid_uw, dimGrid_B, 0, stream1 >> > (height_bfr, width, used_width, d_proc_buff_0, d_proc_buff_1); gpuErrchk(cudaPeekAtLastError()); // 8000 -> 8192
		zero_pad << <dimGrid_uw, dimGrid_B, 0, stream2 >> > (height_bfr, width, used_width, d_proc_buff_3, d_proc_buff_4); gpuErrchk(cudaPeekAtLastError()); // 8000 -> 8192

		cufftErrchk(cufftExecC2C(plan_w2, d_proc_buff_1, d_proc_buff_1, CUFFT_INVERSE));
		cufftErrchk(cufftExecC2C(plan_w2_s2, d_proc_buff_4, d_proc_buff_4, CUFFT_INVERSE));


		scale_IFT_x << <dimGrid_w2, dimGrid_B, 0, stream1 >> > (height_bfr, width_2x, w2Recip, d_proc_buff_1); gpuErrchk(cudaPeekAtLastError());
		scale_IFT_x << <dimGrid_w2, dimGrid_B, 0, stream2 >> > (height_bfr, width_2x, w2Recip, d_proc_buff_4); gpuErrchk(cudaPeekAtLastError());

		//print_gpu_data(1, width_2x, d_proc_buff_1, "d_proc_buff_1-2");

		interp.procInterp(height_ba, width_2xba, dimLine_w2xba, dimLine_B, stream1, d_proc_buff_1, d_proc_buff_2); // calibration?
		interp.procInterp(height_ba, width_2xba, dimLine_w2xba, dimLine_B, stream2, d_proc_buff_4, d_proc_buff_5); // calibration?
	}

	//#pragma omp barrier

	if (threadID == 1) {
		//std::cout<<"	thread :"<<threadID<<"	it:"<<it<<"	frames:"<<frames<<std::endl;
		unsigned long long start_odd = (((unsigned long long)height_1fr * width)) * (p.numBgFrame + (((it * 2 + 1) + 4 + 1 - 1)*p.batchFrames));
		if (it < (frames / 2) - 2) {
			if (it % 2 == 0) {

				//data.loadFile(p.fnameData, start_odd, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_3);
				data.loadMem(p.mem_data_ptr, start_odd, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_3);
				gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_3, h_buff_3, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice, stream2));
			}
			else {

				//data.loadFile(p.fnameData, start_odd, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_4);
				data.loadMem(p.mem_data_ptr, start_odd, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_4);
				gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_4, h_buff_4, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice, stream2));
			}
		}
	}
	if (threadID == 2)
	{

		unsigned long long start_even = (((unsigned long long)height_1fr * width)) * (p.numBgFrame + ((it * 2 + 4 + 1 - 1)*p.batchFrames));
		if (it < ((frames + 1) / 2) - 2) {
			if (it % 2 == 0) {
				//data.loadFile(p.fnameData, start_even, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_1);
				data.loadMem(p.mem_data_ptr, start_even, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_1);
				gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_1, h_buff_1, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice, stream1));
			}
			else {
				//data.loadFile(p.fnameData, start_even, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_2);
				data.loadMem(p.mem_data_ptr, start_even, width, p.camStartPixel, p.camEndPixel, height_bfr, h_buff_2);
				gpuErrchk(cudaMemcpyAsync(d_raw_data_buf_2, h_buff_2, (height_bfr * used_width * sizeof(uint16_t)), cudaMemcpyHostToDevice, stream1));
			}

		}
	}

	if (threadID == 0) {

		mult_divide << <dimLine_w2xba, dimLine_B, 0, stream1 >> > (height_ba, width_2xba, data.d_gauss_win, d_proc_buff_2, /*bg.d_bg_mask*/nullptr, d_proc_buff_2); gpuErrchk(cudaPeekAtLastError());
		mult_divide << <dimLine_w2xba, dimLine_B, 0, stream2 >> > (height_ba, width_2xba, data.d_gauss_win, d_proc_buff_5, /*bg.d_bg_mask*/nullptr, d_proc_buff_5); gpuErrchk(cudaPeekAtLastError());
		//print_gpu_data(width_2x, p.batchAscans, bg.d_bg_mask, "bg_mask");//p.batchAscans * width_2x;
		phi_multiply << <dimLine_w2xba, dimLine_B, 0, stream1 >> > (height_ba, width_2xba, disp.d_fphi, d_proc_buff_2, d_proc_buff_1); gpuErrchk(cudaPeekAtLastError());
		phi_multiply << <dimLine_w2xba, dimLine_B, 0, stream2 >> > (height_ba, width_2xba, disp.d_fphi, d_proc_buff_5, d_proc_buff_4); gpuErrchk(cudaPeekAtLastError());

		cufftErrchk(cufftExecC2C(plan_w2, d_proc_buff_1, d_proc_buff_1, CUFFT_INVERSE));
		cufftErrchk(cufftExecC2C(plan_w2_s2, d_proc_buff_4, d_proc_buff_4, CUFFT_INVERSE));

		start = clock();                                                                                                                                                                                  
		if ((this->process_type == "preview" || this->process_type == "fundus") && it == 0)
		{
			int h = previewFFTrawData.size() / width;
			float2 *data = new float2[h*width_2x]();
			gpuErrchk(cudaMemcpy(data, d_proc_buff_1, (h*width_2x * sizeof(float2)), cudaMemcpyDeviceToHost));
			for (size_t i = 0; i < h; i++)
			{
				for (size_t n = 0; n < width; n++)
				{
					previewFFTrawData[i*width + n] = std::abs(data[i*width_2x+ n].x);
				}
			}
			delete[] data;
		}
		/*std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for write fft :	" << clock() - start << "ms" << std::endl;*/

		scale_IFT << <dimGrid_w2, dimGrid_B, 0, stream1 >> > (height_bfr, width_2x, w2Recip, d_proc_buff_1); gpuErrchk(cudaPeekAtLastError());
		scale_IFT << <dimGrid_w2, dimGrid_B, 0, stream2 >> > (height_bfr, width_2x, w2Recip, d_proc_buff_4); gpuErrchk(cudaPeekAtLastError());

		trim_width << <dimGrid_wt, dimGrid_B, 0, stream1 >> > (height_bfr, width_2x, p.startPixel, p.endPixel, d_proc_buff_1, d_proc_buff_trm); gpuErrchk(cudaPeekAtLastError());
		trim_width << <dimGrid_wt, dimGrid_B, 0, stream2 >> > (height_bfr, width_2x, p.startPixel, p.endPixel, d_proc_buff_4, d_proc_buff_trm_1); gpuErrchk(cudaPeekAtLastError());

		magnitude_db << <dimGrid_wt, dimGrid_B, 0, stream1 >> > (height_bfr, width_trm, d_proc_buff_trm, d_proc_buff_db); gpuErrchk(cudaPeekAtLastError());
		magnitude_db << <dimGrid_wt, dimGrid_B, 0, stream2 >> > (height_bfr, width_trm, d_proc_buff_trm_1, d_proc_buff_db_1); gpuErrchk(cudaPeekAtLastError());

		subt_divide << <dimLine_wtba, dimLine_B, 0, stream1 >> > (height_ba, (width_trm*p.batchAscans), d_proc_buff_db, p.bg_noise, grayRecip, d_proc_buff_db); gpuErrchk(cudaPeekAtLastError());
		subt_divide << <dimLine_wtba, dimLine_B, 0, stream2 >> > (height_ba, (width_trm*p.batchAscans), d_proc_buff_db_1, p.bg_noise, grayRecip, d_proc_buff_db_1); gpuErrchk(cudaPeekAtLastError());

		
		int write_offset_even = (it * 2 * height_bfr*width_trm);
		int write_offset_odd = ((it * 2 + 1)*height_bfr*width_trm);
		multiframe_transpose << <dimGrid_wt, dimGrid_B, 0, stream1 >> > (height_1fr, width_trm, p.batchFrames, d_proc_buff_db, largeBuff + write_offset_even); gpuErrchk(cudaPeekAtLastError());
		multiframe_transpose << <dimGrid_wt, dimGrid_B, 0, stream2 >> > (height_1fr, width_trm, p.batchFrames, d_proc_buff_db_1, largeBuff + write_offset_odd); gpuErrchk(cudaPeekAtLastError());
	}

	if (it % 2 == 0)
	{
		cudaStreamSynchronize(stream2);
	}
	else {
		cudaStreamSynchronize(stream1);
	}

#pragma omp barrier
}

void Output::writeToDisk(float *proc_data_array, std::string datatype, std::vector<uint8_t>& bmpRawData)
{
	cv::Mat image;
	cv::Mat new_image;
	cv::Mat avg_Bscanframe;
	char fname_opencv[200];
	string path_string;

	cv::Mat row_mean;
	cv::Mat avg_img;
	//crop 10% of the x direction
	unsigned int start_x = unsigned int(this->p.numAscansPerBscan*p.crop_percent);
	//make sure start_x equals to 0, it will work properly
	start_x = start_x == 0 ? 1 : start_x;
	// crop 50 on both top and bottom
	cv::Rect re(start_x - 1, 0, this->p.numAscansPerBscan - start_x + 1, this->p.endPixel - p.startPixel + 1 - 140);
	if (datatype == "preview")
	{
		bmpRawData.resize((re.width)*(re.height)*p.numBScans);
	}
	else if (datatype == "fundus")
	{
		bmpRawData.resize(this->p.numAscansPerBscan*(this->frames_tot - 1) + p.numAscansPerBscan * width_trm * 2);
	}
	else
	{
		this->tiffhandler = UltraOCT::TiffHandler(this->p.iop);
	}
	//use tiff
	//TIFF *out;

	//simply adjust by tao to skip the first frame of each scan from safe point
	bool skip_flag = false;
	if (p.avgBscan != 1 && (this->frames_tot == p.batchFrames*p.avgBscan * 1 || this->frames_tot == p.batchFrames*p.avgBscan * 2))
	{
		//means should skip the first frame per avgBscan
		skip_flag = true;
	}

	if (!p.reslice)
	{
		std::cout << "Debug: " << __FUNCTION__ << "frames_tot:" << frames_tot << std::endl;

		unsigned slash_split = p.fnameData.find_last_of("/\\");
		std::cout << "	raw name:" << p.fnameData << std::endl;
		string path_name = p.fnameData.substr(0, slash_split);
		string f_name = p.fnameData.substr(slash_split + 1);
		unsigned period_split = f_name.find_last_of(".");
		string r_name = f_name.substr(0, period_split);
		if (datatype != "acquire")
		{
			std::cout << "	" << r_name << std::endl;
			path_string = (path_name + "\\" + r_name);
			path_string = (path_name + "\\GPU_processed");
			CreateDirectory(path_string.c_str(), NULL);
			path_string = (path_string + "\\" + r_name);
			CreateDirectory(path_string.c_str(), NULL);
		}
		else
		{

			path_string = path_name;
		}
		//tiff class
		string tiff_path = path_string + "\\" + r_name + ".tiff";
		/*string tiff_fundus = path_string + "\\" + r_name + "-1.tiff";*/
		string pseudo_path = path_string + "\\" + r_name + ".png";
		//MagickTiff::TiffWriter tiffwriter(tiff_path);
		bool retiff = false;//this->tiffhandler.OpenTiffFile(tiff_path.c_str());
		if (datatype != "preview" && datatype != "fundus")
		{
			while (!retiff)
			{
				std::cout << __TIMESTAMP__ << "\t" << __FUNCTION__ << "\t" << "can not open file: " << tiff_path << std::endl;
				retiff = this->tiffhandler.OpenTiffFile(tiff_path.c_str());
			}
		}

		if (datatype == "fundus")
		{
			/*this->pageSize.width = this->p.numAscansPerBscan;
			this->pageSize.height = this->frames_tot;
			this->tiffhandler.SetIFDandTiffParams(this->pageSize, 8, 1);*/

			// fundus view result in one image
			for (int i = 0; i < frames_tot-1; i++)
			{
				image = cv::Mat(width_trm, p.numAscansPerBscan, CV_32F, &proc_data_array[i*width_trm*p.numAscansPerBscan]);
				new_image = cv::Mat(width_trm, p.numAscansPerBscan, CV_8U);
				image.convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);

				reduce(new_image, row_mean, 0, CV_REDUCE_AVG);

				memcpy(&bmpRawData[0] + i*p.numAscansPerBscan, row_mean.data, p.numAscansPerBscan);
				//avg_img.push_back(row_mean);
			}
			//XZ
			int middle_frame = (int)((frames_tot - 1) / 2);
			image = cv::Mat(width_trm, p.numAscansPerBscan, CV_32F, &proc_data_array[middle_frame*width_trm*p.numAscansPerBscan]);
			new_image = cv::Mat(width_trm, p.numAscansPerBscan, CV_8U);
			image.convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);
			flip(new_image, image, 0);
			memcpy(&bmpRawData[0] + (frames_tot - 1)*p.numAscansPerBscan, image.data, p.numAscansPerBscan * width_trm);
			image.release();
			new_image.release();
			//YZ
			image = cv::Mat(width_trm, p.numAscansPerBscan, CV_32F, &proc_data_array[(frames_tot - 1)*width_trm*p.numAscansPerBscan]);
			new_image = cv::Mat(width_trm, p.numAscansPerBscan, CV_8U);
			image.convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);
			flip(new_image, image, 0);
			memcpy(&bmpRawData[0] + (frames_tot - 1)*p.numAscansPerBscan + p.numAscansPerBscan * width_trm, image.data, p.numAscansPerBscan * width_trm);
			/*this->tiffhandler.writePage(avg_img.data, 0);
			this->tiffhandler.CloseTiffFile();*/
		}
		else
		{
			if (datatype != "preview")
			{
				this->pageSize.width = re.size().width;
				this->pageSize.height = re.size().height;
				this->tiffhandler.SetIFDandTiffParams(this->pageSize, 8, this->p.numBScans);
			}
			// process the other data type
			int avg_inter = 0;
			int avgBscan = p.avgBscan;
			if (skip_flag)
			{
				--avgBscan;
			}
			avg_Bscanframe = cv::Mat::zeros(width_trm, height_1fr, CV_32F);

			std::cout << "start writing to disk!" << std::endl;
			std::vector<cv::Mat> mat_vec;
			int pageNum = 0;
			for (int i = 0; i < frames_tot; i++)
			{
				if (skip_flag && i%p.avgBscan == 0)
				{
					//skip the frist frame for each avgBscan
					continue;
				}
				// This conversion makes the images look like they do in MATLAB. Doing normalization, like in the
				// image class, results on low contrast images. This needs to be confirmed, but I believe all values below 0 are set to 0.

				//std::cout<<"Debug: "<<__FUNCTION__<<"frames_tot: "<<i<< " avg_inter:" << avg_inter << std::endl;
				/*for (size_t i = 0; i < height_1fr; i++)
				{
					this->edgeCorrect(&proc_data_array[i*width_trm], this->diffs[i], width_trm);
				}*/
				image = cv::Mat(width_trm, height_1fr, CV_32F, &proc_data_array[i*width_trm*height_1fr]);
				cv::Mat dst; 
				//cout << "image = "<< endl << " "  << image << endl << endl;
				//add by tao @ 07/28/2017 to add a median filter to image with the kernel size being odd 3

				if (p.median_filter_size > 1)
				{
					medianBlur(image, dst, p.median_filter_size);
				}
				else
				{
					dst = image;
				}
				if (skip_flag)
				{
					mat_vec.push_back(dst);
				}
				else
				{
					avg_Bscanframe += dst * (1.0 / avgBscan);
				}
				dst.release();

				if (avg_inter == avgBscan - 1) {
					new_image = cv::Mat(width_trm, height_1fr, CV_8U);
					if (skip_flag)
					{
						ofstream outfile("../log/dft_timing.log", ofstream::app);
						clock_t start = clock();
						Output::shiftAndAvg(mat_vec);
						mat_vec[0].convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);
						mat_vec.swap(std::vector<cv::Mat>());
						outfile << __FUNCTION__ << "	" << datatype << ":\t" << "time for phase correlate:	" << clock() - start << " ms" << std::endl;
						outfile.clear();
						outfile.close();
					}
					else
					{
						//avg_Bscanframe.convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);
						//std::cout << "\tpixel:	" << avg_Bscanframe.at<float>(1, 1) << std::endl;
						avg_Bscanframe = avg_Bscanframe * (this->p.contrastA - this->p.contrastB) + this->p.contrastB;
						//avg_Bscanframe.convertTo(new_image, CV_8U, this->p.contrastA, this->p.contrastB);
						avg_Bscanframe.convertTo(new_image, CV_8U, 1, 0);
						avg_Bscanframe.release();
						///cv::normalize(avg_Bscanframe, new_image, 100, 255, cv::NORM_MINMAX, CV_8UC1);
						avg_Bscanframe = cv::Mat::zeros(width_trm, height_1fr, CV_32F);
					}
					//revised by Tao 11/17/2016, adjust the contrast
					//avg_Bscanframe.convertTo(new_image,CV_8U,255);

					flip(new_image, image, 0);
					//add by Tao 12012016

					//for examination occasionally
					//this->CorrectDiff(image);

					if (this->p.flat)
					{
						this->flatOut(image);
					}
					cv::Mat ROI = cv::Mat(image, re);
					cv::Mat save_image;
					//gamma adjust
					ROI.convertTo(save_image, CV_32F);
					cv::normalize(save_image, save_image, 1.0, 0.0, cv::NORM_MINMAX);
					cv::pow(save_image, this->p.gamma, new_image);
					cv::normalize(new_image, new_image, 255.0, 0.0, cv::NORM_MINMAX);
					new_image.convertTo(save_image, CV_8UC1);
					//ROI.copyTo(save_image);
					//std::cout << "\t\ttime for gamma:	" << clock() - start << std::endl;
					
					
					if (datatype == "preview")
					{
						memcpy(&bmpRawData[0] + pageNum * (re.width) * (re.height), save_image.data, (re.width) * (re.height));
					}
					else
					{
						this->tiffhandler.writePage(save_image.data, pageNum);
					}
					++pageNum;
					//std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for write pages :	" << clock() - start << "ms" << std::endl;

					image.release();
					new_image.release();
					save_image.release();
					ROI.release();
				}

				avg_inter = (avg_inter + 1) % avgBscan;

			}
			if (datatype != "preview")
			{
				this->tiffhandler.CloseTiffFile();
			}
			std::cout << "write to tiff finished!" << endl;
		}
	}
	else
	{// reslice before saving to disk
		float *resliced_data_array = new float[height_bfr*width_trm*frames];

		for (int i = 0; i < width_trm; i++)
			for (int j = 0; j < frames_tot; j++)
				for (int k = 0; k < height_1fr; k++)
					resliced_data_array[i*frames_tot*height_1fr + j*height_1fr + k] =
					proc_data_array[j*width_trm*height_1fr + i*height_1fr + k];

		for (int i = 0; i < width_trm; i++)
		{
			image = cv::Mat(frames_tot, height_1fr, CV_32F, &resliced_data_array[i*frames_tot*height_1fr]);
			new_image = cv::Mat(frames_tot, height_1fr, CV_8U);
			image.convertTo(new_image, CV_8U, 255);
			//new_image = cv::Mat(width_trm, height_1fr, CV_16U);
			//image.convertTo(new_image,CV_16U,65535);

			cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
			cv::imshow("Preview", new_image);
			cv::waitKey(1);

			// Try to create the directory on the first iteration.
			if (i == 0)
			{
				unsigned slash_split = p.fnameData.find_last_of("/");
				string path_name = p.fnameData.substr(0, slash_split);
				string f_name = p.fnameData.substr(slash_split + 1);
				unsigned period_split = f_name.find_last_of(".");
				string r_name = f_name.substr(0, period_split);
				//string path_string = (path_name+"\\"+r_name+" - reslice");

				path_string = (path_name + "/GPU_processed");
				CreateDirectory(path_string.c_str(), NULL);
				path_string = (path_string + "/" + r_name + " - reslice");
				CreateDirectory(path_string.c_str(), NULL);
			}

			sprintf(fname_opencv, "%s/%i.tiff", path_string.c_str(), i);
			//cv::imwrite(fname_opencv, new_image);
		}
		cv::destroyWindow("Preview");
		delete[] resliced_data_array;
	}
	image.release();
	new_image.release();
	avg_Bscanframe.release();
	row_mean.release();
	avg_img.release();
	
	cv::waitKey(1);
}


void Output::freeResources()
{
	// modify here for cuda error
	cufftDestroy(plan_w);
	cufftDestroy(plan_w2);
	cufftDestroy(plan_w_s2);
	cufftDestroy(plan_w2_s2);

	gpuErrchk(cudaStreamDestroy(stream1));
	gpuErrchk(cudaStreamDestroy(stream2));
	gpuErrchk(cudaFree(largeBuff));
	gpuErrchk(cudaFree(d_raw_data_buf_1));
	gpuErrchk(cudaFree(d_raw_data_buf_2));
	gpuErrchk(cudaFree(d_raw_data_buf_3));
	gpuErrchk(cudaFree(d_raw_data_buf_4));
	gpuErrchk(cudaFree(d_proc_buff_0));
	gpuErrchk(cudaFree(d_proc_buff_trm));
	gpuErrchk(cudaFree(d_proc_buff_trm_1));
	gpuErrchk(cudaFree(d_proc_buff_db));
	gpuErrchk(cudaFree(d_proc_buff_db_1));
	gpuErrchk(cudaFree(d_proc_buff_trns));
	gpuErrchk(cudaFree(d_proc_buff_trns_1));
	gpuErrchk(cudaFree(d_proc_buff_1));
	gpuErrchk(cudaFree(d_proc_buff_2));
	gpuErrchk(cudaFree(d_proc_buff_3));
	gpuErrchk(cudaFree(d_proc_buff_4));
	gpuErrchk(cudaFree(d_proc_buff_5));

	gpuErrchk(cudaFreeHost(h_buff_1));
	gpuErrchk(cudaFreeHost(h_buff_2));
	gpuErrchk(cudaFreeHost(h_buff_3));
	gpuErrchk(cudaFreeHost(h_buff_4));
}
