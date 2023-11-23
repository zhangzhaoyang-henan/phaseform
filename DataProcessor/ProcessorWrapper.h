#pragma once
#include "prcssr.h"
class ProcessorWrapper
{
private:
	Prcssr *myPrcssr;

	int batchFrames;
	int batchAscans;
	int prevFreq;
	bool reslice;

	int numAscansPerBscan;
	int numBScans;
	int avgBscan;
	int numCameraPixels;
	int numBgFrame;
	int startPixel;
	int endPixel;

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	int cameraStartPixel;
	int cameraEndPixel;
	int numUsedPixels;

	//revised by Tao 11/17/2016
	float alpha;
	int contrastA;
	int contrastB;

	int grayLevel;
	float gamma;
	bool interpLinear;

	bool dispModeGS;
	int dispFrame;
	int dispTests;
	float dispA2;
	float dispA3;

	std::string fnamePhase;
	std::string fnameBg;
	std::string fnameData;

	float crop_percent;
	int median_filter_size;
	float bg_noise, bg_mask_correct;

	int returnValue;
public:
	ProcessorWrapper();
	~ProcessorWrapper();

	void initDLL();

	int param();

	int data();

	int interp();

	int disp();
	int disp(int dispModeGS, float &a2, float &a3);
	int disp(int dispModeGS, float &a2, float &a3, uint16_t *mem_init_data_ptr);

	int bg();
	int bg(uint16_t *mem_bg_ptr);

	int output();

	int output(std::string str, std::string datatype);

	int output(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData);

	// Errors if this conversion is done directly. Some sort of bug in vs2010, workaround with method.
	// Not a bug anymore? Not sure, can't find the stack overflow post. I'm leaving it wrapped in the function.

	std::string sysStrToStdStr(std::string str);

	// When batching directories, we need to recalculate the background for each folder. To do this, the background
	//  class needs to be freed and recreated. This function handles that, passing in the path to the new background file.
	int freeBg(std::string str);

	void destroy();
	// Turn the error string into an error code which the GUI understands.
	void runErrorCheck(std::exception e, int &rv);

	// getters and setters.
	void setBatchFrames(int batchFrames);
	void setBatchAscans(int batchAscans);
	void setPrevFreq(int prevFreq);
	void setReslice(bool reslice);

	void setNumAscansPerBscan(int numAscansPerBscan);
	void setNumBScans(int numBScans);

	void setavgBscan(int avgBscan);

	void setNumCameraPixels(int numCameraPixels);
	void setNumBgFrame(int numBgFrame);
	void setStartPixel(int startPixel);
	void setEndPixel(int endPixel);

	//revised by Tao 11/17/2016
	void setContrastA(int contrastA);
	void setContrastB(int contrastB);
	void setAlpha(float alpha);

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	void setCameraStartPixel(int cameraStartPixel);
	void setCameraEndPixel(int cameraEndPixel);
	void setNumUsedPixels(int numUsedPixels);

	void setGrayLevel(int grayLevel);
	void setGamma(float gamma);
	void setInterpLinear(bool interpLinear);

	void setDispModeGS(bool dispModeGS);
	void setDispFrame(int dispFrame);
	void setDispTests(int dispTests);
	void setDispA2(float dispA2);
	void setDispA3(float dispA3);

	void setFnamePhase(std::string fnamePhase);
	void setFnameBg(std::string fnameBg);
	void setFnameData(std::string fnameData);

	void setIOPofTiffWrapper(const UltraOCT::IOP iop);
	void setMemData(uint16_t *mem_data_ptr);
	void setMemBg(uint16_t *mem_bg_ptr);
	void setGrayDynamic(int gray);
	void setGammaDynamic(float gamma);
	//dynamically set the path to store the data
	void setDataPath(std::string fnameData);
	void setBgNoise(float bg_noise);
	void setBgMaskCorrect(float bg_mask_correct);
	//dynamically set the flat flag to store the data
	void setFlatFlag(bool flat);
	void setA2AndA3(float a2, float a3);
	//release disp and bg for redo
	void releaseDisp()
	{
		this->myPrcssr->releaseDisp();
	}
	void releaseBg()
	{
		this->myPrcssr->releaseBg();
	}

	int getBatchFrames();
	int getBatchAscans();
	int getPrevFreq();
	bool getReslice();

	int getNumAscansPerBscan();
	int getNumBScans();
	int getNumCameraPixels();
	int getNumBgFrame();
	int getStartPixel();
	int getEndPixel();
	float getAlpha();
	int getGrayLevel();
	bool getInterpLinear();

	bool getDispModeGS();
	int getDispFrame();
	int getDispTests();
	float getDispA2();
	float getDispA3();

	std::string getFnamePhase();
	std::string getFnameBg();
	std::string getFnameData();

	void setCropPercent(float crop_percent);
	void setMedianFilterSize(int median_filter_size);

};

