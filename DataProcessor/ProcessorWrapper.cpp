#include "ProcessorWrapper.h"
#include <string>
using namespace std;

ProcessorWrapper::ProcessorWrapper()
{
	this->myPrcssr = nullptr;
}


ProcessorWrapper::~ProcessorWrapper()
{
}


void ProcessorWrapper::initDLL()
{
	myPrcssr = new Prcssr(batchFrames, batchAscans,
		prevFreq, reslice,
		numAscansPerBscan,
		numBScans, avgBscan,
		numCameraPixels, numUsedPixels, cameraStartPixel, cameraEndPixel,
		numBgFrame,
		startPixel, endPixel,
		alpha, grayLevel, gamma, 
		contrastA, contrastB,
		interpLinear, dispModeGS,
		dispFrame, dispTests,
		dispA2, dispA3,
		fnamePhase, fnameData,
		crop_percent, median_filter_size,
		bg_noise, bg_mask_correct
		);

	returnValue = 0;
}

int ProcessorWrapper::param()
{
	try { myPrcssr->runParam(); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::data()
{
	try { myPrcssr->runData(); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::interp()
{
	try { myPrcssr->runInterp(); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::disp()
{
	try { myPrcssr->runDisp(); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::disp(int dispModeGS, float &a2, float &a3)
{
	try { myPrcssr->runDisp(dispModeGS, a2, a3); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::disp(int dispModeGS, float &a2, float &a3, uint16_t *mem_init_data_ptr)
{
	try { myPrcssr->runDisp(dispModeGS, a2, a3, mem_init_data_ptr); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}


int ProcessorWrapper::bg()
{
	try { myPrcssr->runBg(); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::bg(uint16_t *mem_bg_ptr)
{
	try { myPrcssr->runBg(mem_bg_ptr); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::output()
{
	try { myPrcssr->runOutput(); }
	catch (std::exception &ex)
	{
		//DP_LOGGER_ERROR << __FUNCTION__ << " output Error";
		std::cout << __FUNCTION__ << " output Error";
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::output(std::string str, std::string datatype)
{
	string cppStr = sysStrToStdStr(str);
	try { myPrcssr->runOutput(cppStr, datatype); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

int ProcessorWrapper::output(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData)
{
	try 
	{ 
		myPrcssr->runOutput(mem_data_ptr, datatype, bmpRawData, previewFFTrawData);
	}
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

// Errors if this conversion is done directly. Some sort of bug in vs2010, workaround with method.
// Not a bug anymore? Not sure, can't find the stack overflow post. I'm leaving it wrapped in the function.

string ProcessorWrapper::sysStrToStdStr(std::string str)
{
	//string cppString = msclr::interop::marshal_as<std::string>(str);
	string cppString = str;
	return cppString;
}

// When batching directories, we need to recalculate the background for each folder. To do this, the background
//  class needs to be freed and recreated. This function handles that, passing in the path to the new background file.
int ProcessorWrapper::freeBg(std::string str)
{
	string cppStr = sysStrToStdStr(str);
	try { myPrcssr->freeBg(cppStr); }
	catch (std::exception &ex)
	{
		runErrorCheck(ex, returnValue);
	}
	return returnValue;
}

void ProcessorWrapper::destroy()
{
	returnValue = 0;
	if (myPrcssr)
	{
		delete myPrcssr;
		myPrcssr = nullptr;
	}
}

// Turn the error string into an error code which the GUI understands.
void ProcessorWrapper::runErrorCheck(std::exception e, int &rv)
{
	std::cout << "Debug: " << __FUNCTION__ << " " << rv << std::endl;
	if (e.what() == std::string("Calibration File Error"))
	{
		rv = 1;
		//DP_LOGGER_ERROR << __FUNCTION__ << " Calibration File Error";
		std::cout << __FUNCTION__ << " Calibration File Error";
	}
	else if (e.what() == std::string("File Open Error"))
	{
		rv = 2;
		//DP_LOGGER_ERROR << __FUNCTION__ << " File Open Error";
		std::cout << __FUNCTION__ << " File Open Error";
	}
	else if (e.what() == std::string("; File Error"))
	{
		rv = 3;
		//DP_LOGGER_ERROR << __FUNCTION__ << " File Error";
		std::cout << __FUNCTION__ << " ile Error";
	}
	else if (e.what() == std::string("Gridsearch Error"))
	{
		rv = 4;
		//DP_LOGGER_ERROR << __FUNCTION__ << " Gridsearch Error";
		std::cout << __FUNCTION__ << " Gridsearch Error";
	}
	else if (e.what() == std::string("CUDA Error"))
	{
		rv = 5;
		//DP_LOGGER_ERROR << __FUNCTION__ << " CUDA Error";
		std::cout << __FUNCTION__ << " CUDA Error";
	}
	else if (e.what() == std::string("CUFFT Error"))
	{
		rv = 6;
		//DP_LOGGER_ERROR << __FUNCTION__ << " CUFFT Error";
		std::cout << __FUNCTION__ << " CUFFT Error";
	}
	else if (e.what() == std::string("ROI Error"))
	{
		rv = 9;
		//DP_LOGGER_ERROR << __FUNCTION__ << " ROI Error";
		std::cout << __FUNCTION__ << " ROI Error";
	}
	else
	{
		rv = 7;
		//DP_LOGGER_ERROR << __FUNCTION__ << " Unknown Error";
		std::cout << __FUNCTION__ << " Unknown Error";
	}
}

// getters and setters.
void ProcessorWrapper::setBatchFrames(int batchFrames){ this->batchFrames = batchFrames; }
void ProcessorWrapper::setBatchAscans(int batchAscans){ this->batchAscans = batchAscans; }
void ProcessorWrapper::setPrevFreq(int prevFreq){ this->prevFreq = prevFreq; }
void ProcessorWrapper::setReslice(bool reslice){ this->reslice = reslice; }

void ProcessorWrapper::setNumAscansPerBscan(int numAscansPerBscan){ this->numAscansPerBscan = numAscansPerBscan; }
void ProcessorWrapper::setNumBScans(int numBScans){ this->numBScans = numBScans; }

void ProcessorWrapper::setavgBscan(int avgBscan){ this->avgBscan = avgBscan; }

void ProcessorWrapper::setNumCameraPixels(int numCameraPixels){ this->numCameraPixels = numCameraPixels; }
void ProcessorWrapper::setNumBgFrame(int numBgFrame){ this->numBgFrame = numBgFrame; }
void ProcessorWrapper::setStartPixel(int startPixel){ this->startPixel = startPixel; }
void ProcessorWrapper::setEndPixel(int endPixel){ this->endPixel = endPixel; }

//revise by Tao 11/17/2016
void ProcessorWrapper::setContrastA(int contrastA){ this->contrastA = contrastA; }
void ProcessorWrapper::setContrastB(int contrastB){ this->contrastB = contrastB; }
void ProcessorWrapper::setAlpha(float alpha){ this->alpha = alpha; }

// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
void ProcessorWrapper::setCameraStartPixel(int cameraStartPixel){ this->cameraStartPixel = cameraStartPixel; }
void ProcessorWrapper::setCameraEndPixel(int cameraEndPixel){ this->cameraEndPixel = cameraEndPixel; }
void ProcessorWrapper::setNumUsedPixels(int numUsedPixels){ this->numUsedPixels = numUsedPixels; }

void ProcessorWrapper::setGrayLevel(int grayLevel){ this->grayLevel = grayLevel; }
void ProcessorWrapper::setGamma(float gamma){ this->gamma = gamma; }
void ProcessorWrapper::setGammaDynamic(float gamma){ this->myPrcssr->setGammaDynamic(gamma); }
void ProcessorWrapper::setInterpLinear(bool interpLinear){ this->interpLinear = interpLinear; }

void ProcessorWrapper::setDispModeGS(bool dispModeGS){ this->dispModeGS = dispModeGS; }
void ProcessorWrapper::setDispFrame(int dispFrame){ this->dispFrame = dispFrame; }
void ProcessorWrapper::setDispTests(int dispTests){ this->dispTests = dispTests; }
void ProcessorWrapper::setDispA2(float dispA2){ this->dispA2 = dispA2; }
void ProcessorWrapper::setDispA3(float dispA3){ this->dispA3 = dispA3; }

void ProcessorWrapper::setFnamePhase(std::string fnamePhase){ this->fnamePhase = fnamePhase; }
void ProcessorWrapper::setFnameBg(std::string fnameBg){ this->fnameBg = fnameBg; }
void ProcessorWrapper::setFnameData(std::string fnameData){ this->fnameData = fnameData; }

void ProcessorWrapper::setIOPofTiffWrapper(const UltraOCT::IOP iop)
{
	this->myPrcssr->setIOPofTiff(iop);
}

void ProcessorWrapper::setMemData(uint16_t *mem_data_ptr)
{
	this->myPrcssr->setMemData(mem_data_ptr);
}

void ProcessorWrapper::setMemBg(uint16_t *mem_bg_ptr)
{
	this->myPrcssr->setMemBg(mem_bg_ptr);
}

void ProcessorWrapper::setGrayDynamic(int gray)
{
	this->myPrcssr->setGrayDynamic(gray);
}

void ProcessorWrapper::setDataPath(std::string fnameData)
{
	this->myPrcssr->setDataPath(fnameData);
}
void ProcessorWrapper::setFlatFlag(bool flat)
{
	this->myPrcssr->setFlatFlag(flat);
}
void ProcessorWrapper::setA2AndA3(float a2, float a3)
{
	this->myPrcssr->setA2AndA3(a2, a3);
}
void ProcessorWrapper::setBgNoise(float bg_noise) { this->bg_noise = bg_noise; }
void ProcessorWrapper::setBgMaskCorrect(float bg_mask_correct) { this->bg_mask_correct = bg_mask_correct; }
int ProcessorWrapper::getBatchFrames(){ return batchFrames; }
int ProcessorWrapper::getBatchAscans(){ return batchAscans; }
int ProcessorWrapper::getPrevFreq(){ return prevFreq; }
bool ProcessorWrapper::getReslice(){ return reslice; }

int ProcessorWrapper::getNumAscansPerBscan(){ return numAscansPerBscan; }
int ProcessorWrapper::getNumBScans(){ return numBScans; }
int ProcessorWrapper::getNumCameraPixels(){ return numCameraPixels; }
int ProcessorWrapper::getNumBgFrame(){ return numBgFrame; }
int ProcessorWrapper::getStartPixel(){ return startPixel; }
int ProcessorWrapper::getEndPixel(){ return endPixel; }
float ProcessorWrapper::getAlpha(){ return alpha; }
int ProcessorWrapper::getGrayLevel(){ return grayLevel; }
bool ProcessorWrapper::getInterpLinear(){ return interpLinear; }

bool ProcessorWrapper::getDispModeGS(){ return dispModeGS; }
int ProcessorWrapper::getDispFrame(){ return dispFrame; }
int ProcessorWrapper::getDispTests(){ return dispTests; }
float ProcessorWrapper::getDispA2(){ return dispA2; }
float ProcessorWrapper::getDispA3(){ return dispA3; }

std::string ProcessorWrapper::getFnamePhase(){ return fnamePhase; }
std::string ProcessorWrapper::getFnameBg(){ return fnameBg; }
std::string ProcessorWrapper::getFnameData(){ return fnameData; }

void ProcessorWrapper::setCropPercent(float crop_percent) { this->crop_percent = crop_percent; }
void ProcessorWrapper::setMedianFilterSize(int median_filter_size) { this->median_filter_size = median_filter_size; }