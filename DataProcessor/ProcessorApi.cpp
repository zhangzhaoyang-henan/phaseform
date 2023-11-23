#include "ProcessorApi.h"
#include <fstream>
#include <iostream>
#include <time.h>
#include <Magick++.h>

ProcessorApi::ProcessorApi()
{
	prcsrCali = new ProcessorCalibration();
	octPrcssr = new ProcessorWrapper();
}


ProcessorApi::~ProcessorApi()
{
	if (prcsrCali != NULL)
	{
		delete prcsrCali; prcsrCali = NULL;
	}
	if (octPrcssr != NULL)
	{
		octPrcssr->destroy();
		delete octPrcssr;
		octPrcssr = NULL;
	}
}

// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
void ProcessorApi::setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
	int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
	int startPixel, int endPixel, int grayLevel, bool interpLinear,
	bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
	std::string fnamePhase, std::string fnameBg, std::string fnameData,
	int contrastA, int contrastB,
	float crop_percent, int median_filter_size)
{
	prcsrCali->setProcessorCalibration(batchFrames, batchAscans, prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan, numCameraPixels, numBgFrame,
		cameraStartPixel, cameraEndPixel, numUsedPixels,
		startPixel, endPixel, grayLevel, interpLinear,
		dispModeGS, dispFrame, dispTests, dispA2, dispA3,
		fnamePhase, fnameBg, fnameData,
		contrastA, contrastB,
		crop_percent, median_filter_size);
	setPrames();
}

// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
void ProcessorApi::setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
	int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
	int startPixel, int endPixel, int grayLevel, bool interpLinear,
	bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
	std::string fnamePhase, std::string fnameData,
	int contrastA, int contrastB,
	float crop_percent, int median_filter_size)
{
	prcsrCali->setProcessorCalibration(batchFrames, batchAscans, prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan, numCameraPixels, numBgFrame,
		cameraStartPixel, cameraEndPixel, numUsedPixels,
		startPixel, endPixel, grayLevel, interpLinear,
		dispModeGS, dispFrame, dispTests, dispA2, dispA3,
		fnamePhase, fnameData,
		contrastA, contrastB,
		crop_percent, median_filter_size);
	setPrames();
}

void ProcessorApi::setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
	int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
	int startPixel, int endPixel, int grayLevel, float gamma, bool interpLinear,
	bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
	std::string fnamePhase, std::string fnameData,
	int contrastA, int contrastB, float crop_percent, int median_filter_size,
	float bg_noise, float bg_mask_correct)
{
	prcsrCali->setProcessorCalibration(batchFrames, batchAscans, prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan, numCameraPixels, numBgFrame,
		cameraStartPixel, cameraEndPixel, numUsedPixels,
		startPixel, endPixel, grayLevel, gamma, interpLinear,
		dispModeGS, dispFrame, dispTests, dispA2, dispA3,
		fnamePhase, fnameData,
		contrastA, contrastB,
		crop_percent, median_filter_size,
		bg_noise, bg_mask_correct);
	setPrames();
}

void ProcessorApi::resetDispersion(bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3)
{
	prcsrCali->resetDispersion(dispModeGS, dispFrame, dispTests, dispA2, dispA3);
	setPrames();
}

void ProcessorApi::setPrames()
{
	// Send settings
	//std::cout<<"Debug: "<<__FUNCTION__<<"prcsrCali->m_batchFrames: "<<prcsrCali->m_batchFrames<<std::endl;

	octPrcssr->setBatchFrames(prcsrCali->m_batchFrames);
	octPrcssr->setBatchAscans(prcsrCali->m_batchAscans);
	octPrcssr->setPrevFreq(prcsrCali->m_prevFreq);
	octPrcssr->setReslice(prcsrCali->m_reslice);

	octPrcssr->setNumAscansPerBscan(prcsrCali->m_numAscansPerBscan);
	octPrcssr->setNumBScans(prcsrCali->m_numBScans);

	//cout << __FUNCTION__ << "m_numBScans " << prcsrCali->m_numBScans;
	octPrcssr->setavgBscan(prcsrCali->m_avgBscan);

	octPrcssr->setNumCameraPixels(prcsrCali->m_numCameraPixels);
	octPrcssr->setNumBgFrame(prcsrCali->m_numBgFrame);
	octPrcssr->setStartPixel(prcsrCali->m_startPixel);
	octPrcssr->setEndPixel(prcsrCali->m_endPixel);

	//add by Tao 11/17/2016
	octPrcssr->setContrastA(prcsrCali->m_contrastA);
	octPrcssr->setContrastB(prcsrCali->m_contrastB);
	octPrcssr->setAlpha(prcsrCali->m_alpha);

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	octPrcssr->setCameraStartPixel(prcsrCali->m_cameraStartPixel);
	octPrcssr->setCameraEndPixel(prcsrCali->m_cameraEndPixel);
	octPrcssr->setNumUsedPixels(prcsrCali->m_numUsedPixels);

	octPrcssr->setGrayLevel(prcsrCali->m_grayLevel);
	octPrcssr->setGamma(prcsrCali->m_gamma);
	octPrcssr->setInterpLinear(prcsrCali->m_interpLinear);

	// What kind of dispersion compensation was selected?
	octPrcssr->setDispModeGS(prcsrCali->m_dispModeGS);

	octPrcssr->setDispA2(prcsrCali->m_dispA2);
	octPrcssr->setDispA3(prcsrCali->m_dispA3);
	octPrcssr->setDispTests(prcsrCali->m_dispTests);
	octPrcssr->setDispFrame(prcsrCali->m_dispFrame);

	octPrcssr->setFnamePhase(prcsrCali->m_fnamePhase);
	//octPrcssr->setFnameBg(prcsrCali->m_fnameBg);
	octPrcssr->setFnameData(prcsrCali->m_fnameData);
	//octPrcssr->setMemData(prcsrCali->mem_data_ptr);
	//octPrcssr->setMemBg(prcsrCali->mem_bg_ptr);

	octPrcssr->setCropPercent(prcsrCali->m_crop_percent);
	octPrcssr->setMedianFilterSize(prcsrCali->m_median_filter_size);
	octPrcssr->setBgNoise(prcsrCali->m_bg_noise);
	octPrcssr->setBgMaskCorrect(prcsrCali->m_bg_mask_correct);
}

bool ProcessorApi::processBg()
{
	time_t now = time(0);
	tm *ltm = localtime(&now);
	//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
	//cout<<"processbg 0"<<endl;

	int result = 0;
	try
	{
		octPrcssr->initDLL();
		result = octPrcssr->param();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 1"<<endl;

		//std::cout<<"Debug: "<<__FUNCTION__<<" "<<result<<std::endl;
		if (result != 0) { throw std::invalid_argument("param error"); }

		result = octPrcssr->data();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 2"<<endl;

		//std::cout<<"Debug: "<<__FUNCTION__<<" "<<result<<std::endl;
		if (result != 0) { throw std::invalid_argument("data error"); }
		result = octPrcssr->interp();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 3"<<endl;

		if (result != 0) { throw std::invalid_argument("interp error"); }

		result = octPrcssr->disp();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 4"<<endl;

		if (result != 0) { throw std::invalid_argument("disp error"); }
		result = octPrcssr->bg();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 5"<<endl;

		if (result != 0) { throw std::invalid_argument("bg error"); }
		//result = octPrcssr->output();

		now = time(0);
		ltm = localtime(&now);
		//cout<<1+ltm->tm_hour<<":"<<1+ltm->tm_min<<":"<<1 + ltm->tm_sec<<"\t";
		//cout<<"processbg 6"<<endl;

		if (result != 0) { throw std::invalid_argument("output error"); }

		//DP_LOGGER_INFO << __FUNCTION__ << "\t\tprocessBg done.";
		return true;

	}
	catch (std::exception &e)
	{
		std::cout << "Error: " << __FUNCTION__ << " " << result << std::endl;
		//DP_LOGGER_ERROR << __FUNCTION__ << "\t\tprocessBg error:";
		octPrcssr->destroy();
		delete octPrcssr; octPrcssr = NULL;	// Not sure if this does anything
		return false;
	}

}

bool ProcessorApi::processBg(int dispModeGS, float &a2, float &a3)
{
	int result = 0;

	octPrcssr->initDLL();
	result = octPrcssr->param();
	//std::cout<<"Debug: "<<__FUNCTION__<<" param: "<<result<<std::endl;
	if (result != 0) { throw std::invalid_argument("param error"); }

	//cout<<"before data"<<endl;
	result = octPrcssr->data();
	if (result != 0) { throw std::invalid_argument("data error"); }
	result = octPrcssr->interp();
	if (result != 0) { throw std::invalid_argument("interp error"); }

	result = octPrcssr->disp(dispModeGS, a2, a3);
	if (result != 0) { throw std::invalid_argument("disp error"); }
	result = octPrcssr->bg();
	if (result != 0) { throw std::invalid_argument("bg error"); }
	//result = octPrcssr->output();
	//if (result != 0) { throw invalid_argument("output error"); }

	//DP_LOGGER_INFO << __FUNCTION__ << "\t\tprocessBg done.";
	std::cout << __FUNCTION__ << "\t\tprocessBg done." << std::endl;

	//octPrcssr->destroy();
	//delete octPrcssr; octPrcssr = NULL;	// Not sure if this does anything
	return true;
}

bool ProcessorApi::processBg(int dispModeGS, float &a2, float &a3, uint16_t *mem_bg_ptr, uint16_t *mem_data_ptr)
{
	int result = 0;
	octPrcssr->initDLL();
	result = octPrcssr->param();
	if (result != 0) { throw std::invalid_argument("param error"); }
	result = octPrcssr->data();
	if (result != 0) { throw std::invalid_argument("data error"); }
	result = octPrcssr->interp();
	if (result != 0) { throw std::invalid_argument("interp error"); }
	this->setMemData(mem_data_ptr);
	this->setMemBg(mem_bg_ptr);
	//result = octPrcssr->disp(dispModeGS, a2, a3, mem_data_ptr);
	result = octPrcssr->disp(dispModeGS, a2, a3);
	if (result != 0) { throw std::invalid_argument("disp error"); }
	result = octPrcssr->bg();
	std::cout << "result:" << result << std::endl;
	if (result != 0) { throw std::invalid_argument("bg error"); }
	return true;
}

bool ProcessorApi::processBg(int dispModeGS, float &a2, float &a3, uint16_t *mem_data_ptr)
{
	//std::cout << __FUNCTION__ << "\t go into process bg" << std::endl;
	uint16_t *mem_bg_ptr = (uint16_t*)malloc(sizeof(uint16_t) /** this->prcsrCali->m_numAscansPerBscan */
		* this->prcsrCali->m_numCameraPixels /** this->prcsrCali->m_numBScans*/);
	clock_t start = clock();
	this->setBg(mem_bg_ptr, mem_data_ptr);
	std::cout << "		set bg time:" << clock() - start << std::endl;
	//std::cout << __FUNCTION__ << "\t go into setbg" << std::endl;
	int result = 0;

	octPrcssr->initDLL();
	//std::cout << __FUNCTION__ << "\t go into initDll" << std::endl;
	result = octPrcssr->param();
	//std::cout << __FUNCTION__ << "\t go into param" << std::endl;
	if (result != 0) { throw std::invalid_argument("param error"); }
	result = octPrcssr->data();
	//std::cout << __FUNCTION__ << "\t go into data" << std::endl;
	if (result != 0) { throw std::invalid_argument("data error"); }
	result = octPrcssr->interp();
	if (result != 0) { throw std::invalid_argument("interp error"); }
	//std::cout << __FUNCTION__ << "\t go into interp" << std::endl;
	this->setMemData(mem_data_ptr); //赋值
	this->setMemBg(mem_bg_ptr); // 赋值
	//std::cout << __FUNCTION__ << "\t go into setMem" << std::endl;
	result = octPrcssr->disp(dispModeGS, a2, a3);
	if (result != 0) { throw std::invalid_argument("disp error"); }
	//std::cout << __FUNCTION__ << "\t go into setMem" << std::endl;
	result = octPrcssr->bg();
	//std::cout << __FUNCTION__ << "\t go into bg" << std::endl;
	if (result != 0) { throw std::invalid_argument("bg error"); }
	free(mem_bg_ptr);
	//std::cout << __FUNCTION__ << "\t go into free" << std::endl;
	return true;
}

//@wp start 2018/5/4
bool ProcessorApi::setBg(uint16_t *mem_bg, uint16_t *mem_data_ptr)
{
	try
	{
		std::cout << __FUNCTION__ << "\t goes into function setBg" << std::endl;
		//each bscan has its bg
		memset(&mem_bg[0], 0, this->prcsrCali->m_numCameraPixels/**this->prcsrCali->m_numBScans*/ * sizeof(uint16_t));
		std::vector<float> avg(this->prcsrCali->m_numCameraPixels, 0);

		int max_ = 0;
		std::cout << __FUNCTION__ << "\t numBscan:" << this->prcsrCali->m_numBScans << std::endl;
		for (size_t k = 0; k < this->prcsrCali->m_numBScans; k++) //10
		{
			for (size_t i = 0; i < this->prcsrCali->m_numAscansPerBscan; ++i) //1200
			{
				//avg[i] = 0;
				for (size_t j = 0; j < this->prcsrCali->m_numCameraPixels; ++j) //2048
				{
					//std::cout << __FUNCTION__ << "\tcount: " << k*this->prcsrCali->m_numAscansPerBscan*this->prcsrCali->m_numCameraPixels + i * this->prcsrCali->m_numCameraPixels + j << std::endl;
					//avg[i] += mem_data[j * 2048 + i];
					/*avg[k*this->prcsrCali->m_numCameraPixels+j] += mem_data_ptr[k*this->prcsrCali->m_numAscansPerBscan*this->prcsrCali->m_numCameraPixels
						+ i * this->prcsrCali->m_numCameraPixels + j];*/
					avg[j] += mem_data_ptr[k*this->prcsrCali->m_numAscansPerBscan*this->prcsrCali->m_numCameraPixels
						+ i * this->prcsrCali->m_numCameraPixels + j];
					/*mem_bg[i] += mem_data_ptr[j * this->prcsrCali->m_numCameraPixels + i];*/
					//std::cout << "avg" << j << avg[j] << std::endl;
				}
			}
		}
		std::cout << __FUNCTION__ << "\t goes into function setBg for loop" << std::endl;
		for (size_t i = 0; i < this->prcsrCali->m_numCameraPixels; ++i)
		{
			avg[i] /= 1.0*this->prcsrCali->m_numAscansPerBscan*this->prcsrCali->m_numBScans; //avg[i]=avg[i]/(1*1200*10)
			mem_bg[i] = avg[i];
			/*mem_bg[i] /= 1.0*this->prcsrCali->m_numAscansPerBscan;*/
		}
	}
	catch (std::exception &e)
	{
		std::cout << __FUNCTION__ << e.what() << std::endl;
		return false;
	}

	return true;
}

bool ProcessorApi::processFrameData(std::string fnameData2, std::string datatype)
{
	//debug @Tao 10/02/16
	//DP_LOGGER_ERROR << "	fnameData2:" << fnameData2 << "	datatype:" << datatype << endl;

	int result;
	result = octPrcssr->output(fnameData2, datatype);
	if (result != 0) {
		//DP_LOGGER_ERROR << __FUNCTION__ << "\t\tprocessFrameData error.";
		std::cout << __FUNCTION__ << "\t\tprocessFrameData error.";
		throw std::invalid_argument("output error");
	}

	std::cout << __FUNCTION__ << "\t\tprocessFrameData done." << std::endl;

	if (octPrcssr != NULL)
	{
		octPrcssr->destroy();
		std::cout << __FUNCTION__ << "\t\t octPrcssr->destroy(): release cuda resources." << std::endl;
	}

	return true;
}

//add by tao @ 06/21/2017 to process the memory data
bool ProcessorApi::processFrameData(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData)
{

	int result=0;
	result = octPrcssr->output(mem_data_ptr, datatype, bmpRawData, previewFFTrawData);
	if (result != 0) {
		//DP_LOGGER_ERROR << __FUNCTION__ << "\t\tprocessFrameData error.";
		std::cout << __FUNCTION__ << "\t\tprocessFrameData error.";
		throw std::invalid_argument("output error");
	}

	std::cout << __FUNCTION__ << "\t\tprocessFrameData done." << std::endl;

	//set the ptr of data and bg to nullptr
	this->setMemData(nullptr);
	this->setMemBg(nullptr);

	if (octPrcssr != NULL)
	{
		octPrcssr->destroy();
		std::cout << __FUNCTION__ << "\t\t octPrcssr->destroy(): release cuda resources." << std::endl;
	}

	return true;
}

//add by tao @ 01/22/2019
void ProcessorApi::gridSearchA2(uint16_t *mem_data_ptr, float& a2, float& a3, std::string fnameData2, std::string tiff_path, std::string datatype, float step)
{
	std::vector<uint8_t> bmpRawData;
	std::vector<float> tmpvec;
	//float a2 = 0, a3 = 0;
	/*std::vector<uint16_t> mem_bg;*/
	std::cout << "\n" << __FUNCTION__ << "\t start grid search for the best a2 and a3----------------------" << std::endl;
	uint16_t *mem_bg_ptr = (uint16_t*)malloc(sizeof(uint16_t) /** this->prcsrCali->m_numAscansPerBscan */
		* this->prcsrCali->m_numCameraPixels /** this->prcsrCali->m_numBScans*/);


	int result = 0;

	octPrcssr->initDLL();
	result = octPrcssr->param();
	if (result != 0) { throw std::invalid_argument("param error"); }
	this->setBg(mem_bg_ptr, mem_data_ptr);
	this->setMemData(mem_data_ptr);
	this->setMemBg(mem_bg_ptr);
	//cout<<"before data"<<endl;
	result = octPrcssr->data();
	if (result != 0) { throw std::invalid_argument("data error"); }
	result = octPrcssr->interp();
	if (result != 0) { throw std::invalid_argument("interp error"); }


	//debug @Tao 10/02/16
	//DP_LOGGER_ERROR << "	fnameData2:" << fnameData2 << "	datatype:" << datatype << endl;
	this->setDataPath(fnameData2);
	this->setFlatFlag(true);
	double gap = step;
	std::cout << __FUNCTION__ << "\tgap:" << gap << std::endl;
	float trialA2 = a2;
	std::vector<double> gridSearchVals;
	std::vector<double> a2s;
	long long count_i = 0;
	for (long long i = -40; i <= 40; ++i, ++count_i)
	{
		trialA2 = a2 + i * gap;
		this->octPrcssr->setA2AndA3(trialA2, a3);
		this->octPrcssr->releaseDisp();
		//result = octPrcssr->disp(dispModeGS, a2, a3, mem_data_ptr);
		result = octPrcssr->disp(false, trialA2, a3);
		if (result != 0) { throw std::invalid_argument("disp error"); }

		this->octPrcssr->releaseBg();
		result = octPrcssr->bg();
		//result - octPrcssr->bg(mem_bg_ptr);
		if (result != 0) { throw std::invalid_argument("bg error"); }
		//processFrameData(mem_data_ptr, datatype);

		int result;
		result = octPrcssr->output(mem_data_ptr, datatype, bmpRawData, tmpvec);

		if (result != 0) {
			//DP_LOGGER_ERROR << __FUNCTION__ << "\t\tprocessFrameData error.";
			std::cout << __FUNCTION__ << "\t\tprocessFrameData error.";
			throw std::invalid_argument("output error");
		}
		gridSearchVals.push_back(this->get_img_clarity(tiff_path));
		a2s.push_back(trialA2);

		/*std::string oldName = fnameData2.substr(0, fnameData2.size() - 4) + ".tiff";
		std::string newName = fnameData2.substr(0, fnameData2.size() - 4) + "-" + std::to_string(count_i) + ".tiff";
		if (!rename(oldName.c_str(), newName.c_str()))
		{
		std::cout << "rename success " << std::endl;
		}
		else
		{
		std::cout << "rename error " << std::endl;
		}

		std::cout << __FUNCTION__ << "\t\tprocessFrameData done." << std::endl;*/

		//set the ptr of data and bg to nullptr
		std::cout << "	finish: " << i << std::endl;
	}

	std::vector<double>::iterator min_iter;
	min_iter = std::min_element(gridSearchVals.begin(), gridSearchVals.end());
	int argMax = std::distance(gridSearchVals.begin(), min_iter);

	trialA2 = a2s[argMax];
	a2 = trialA2;
	this->octPrcssr->setA2AndA3(trialA2, a3);
	this->octPrcssr->releaseDisp();
	//result = octPrcssr->disp(dispModeGS, a2, a3, mem_data_ptr);
	result = octPrcssr->disp(false, trialA2, a3);
	if (result != 0) { throw std::invalid_argument("disp error"); }

	this->octPrcssr->releaseBg();
	result = octPrcssr->bg();
	//result - octPrcssr->bg(mem_bg_ptr);
	if (result != 0) { throw std::invalid_argument("bg error"); }
	//processFrameData(mem_data_ptr, datatype);

	result = octPrcssr->output(mem_data_ptr, datatype, bmpRawData, tmpvec);

	if (result != 0) {
		//DP_LOGGER_ERROR << __FUNCTION__ << "\t\tprocessFrameData error.";
		std::cout << __FUNCTION__ << "\t\tprocessFrameData error.";
		throw std::invalid_argument("output error");
	}

	this->setMemData(nullptr);
	this->setMemBg(nullptr);
	if (octPrcssr != NULL)
	{
		octPrcssr->destroy();
		std::cout << __FUNCTION__ << "\t\t octPrcssr->destroy(): release cuda resources." << std::endl;
	}
	delete mem_bg_ptr;
}

void ProcessorApi::setIOPofTiffApi(const UltraOCT::IOP iop)
{
	this->octPrcssr->setIOPofTiffWrapper(iop);
}

// add by tao @06/21/2017
void ProcessorApi::setMemData(uint16_t *mem_data_ptr)
{
	octPrcssr->setMemData(mem_data_ptr);
}

void ProcessorApi::setMemBg(uint16_t *mem_bg_ptr)
{
	octPrcssr->setMemBg(mem_bg_ptr);
}

void ProcessorApi::setGrayDynamic(int gray)
{
	//this->octPrcssr->setGrayDynamic(gray);
	this->octPrcssr->setGrayLevel(gray);
}
void ProcessorApi::setGammaDynamic(float gamma)
{
	this->octPrcssr->setGamma(gamma);
}

void ProcessorApi::setContrastADynamic(int contrastA)
{
	this->octPrcssr->setContrastA(contrastA);
}

void ProcessorApi::setDataPath(std::string fnameData)
{
	octPrcssr->setDataPath(fnameData);
}

void ProcessorApi::setFlatFlag(bool flat)
{
	octPrcssr->setFlatFlag(flat);
}

void ProcessorApi::CorrectDiff(std::string path)
{
	Magick::Image magick_image;
	magick_image.read(path + "[" + std::to_string(static_cast<long long>(0)) + "]");
	Magick::Blob blob;
	magick_image.write(&blob);
	cv::Mat srcImage(magick_image.rows(), magick_image.columns(), CV_8UC1);
	//cv::Mat(magick_image.rows(), magick_image.columns(), (void *)blob.data());
	//image.write(0, 0, w, h, "BGR", Magick::CharPixel, opencvImage.data);
	std::memcpy(srcImage.data, blob.data(), magick_image.rows()*magick_image.columns());
	//cv::Mat srcImage = cv::imread(path);


	std::vector<cv::Mat> channels;
	cv::split(srcImage.clone(), channels);
	cv::Mat gray = channels.at(0);
	int height = gray.size().height;
	int width = gray.size().width;
	std::vector<int> diff(width, 0);
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
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::sort(contours.begin(), contours.end(),
		[](std::vector<cv::Point> a, std::vector<cv::Point> b) {return a.size() > b.size(); });


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
	std::ofstream outFile;
	outFile.open("../conf/diff.csv", std::ios::out);
	for (int j = 0; j < width; j++)
	{
		outFile << diff[j] << std::endl;
	}
	for (size_t i = width; i < 4096; ++i)
	{
		outFile << 0 << std::endl;
	}
	outFile.close();
	outFile.clear();
}
void ProcessorApi::blurDiff(std::vector<int> &diff)
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

void ProcessorApi::WrappercudaDeviceReset()
{
	cudaDeviceReset();
	std::cout << __FUNCTION__ << "\t" << __DATE__ << "\t" << __TIME__ << "\t Reset Cuda Devices" << std::endl;
}

double ProcessorApi::get_img_clarity(std::string path)
{
	try
	{
		cv::Mat src = cv::imread(path, 0);
		cv::Mat img;
		cv::threshold(src, img, 190, 255, cv::THRESH_TOZERO);
		/*cv::namedWindow("threshold", CV_WINDOW_NORMAL);
		cv::imshow("threshold", img);
		cv::waitKey();
		cv::destroyWindow("threshold");*/
		double P = 0;
		for (int i = 1; i < img.rows - 1; i++)
		{
			//定义行指针
			uchar *current_ptr = (uchar*)img.data + i * img.cols;
			uchar *pre_ptr = (uchar*)img.data + (i - 1)*img.cols;
			uchar *next_ptr = (uchar*)img.data + (i + 1)*img.cols;
			for (int j = 1; j < img.cols - 1; j++)
			{
				P += abs((pre_ptr[j - 1] - current_ptr[j])*0.7 + pre_ptr[j] - current_ptr[j] + (pre_ptr[j + 1] - current_ptr[j])*0.7
					+ (current_ptr[j - 1] - current_ptr[j]) + current_ptr[j + 1] - current_ptr[j]
					+ (next_ptr[j - 1] - current_ptr[j])*0.7 + next_ptr[j] - current_ptr[j] + (next_ptr[j + 1] - current_ptr[j])*0.7);

			}
		}
		return P / (img.cols - 2) / (img.rows - 2);
	}
	catch (std::exception& e)
	{
		throw e;
	}
}