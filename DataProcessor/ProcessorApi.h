#pragma once


#include "ProcessorCalibration.h"
#include "ProcessorWrapper.h"
class ProcessorApi
{
private:
	ProcessorCalibration* prcsrCali;
	ProcessorWrapper* octPrcssr;
	void setPrames();
public:
	ProcessorApi();
	~ProcessorApi();

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameBg, std::string fnameData,
		int contrastA, int contrastB,
		float crop_percent, int median_filter_size);
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		int contrastA, int contrastB,
		float crop_percent, int median_filter_size);
	//add bg noise and bg_mask
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, float gamma, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		int contrastA, int contrastB, float crop_percent, int median_filter_size,
		float bg_noise, float bg_mask_correct);
	void resetDispersion(bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3);

	bool processBg();
	bool processBg(int dispModeGS, float &a2, float &a3);
	bool processBg(int dispModeGS, float &a2, float &a3, uint16_t *mem_bg_ptr, uint16_t *mem_data_ptr);
	//override: use avg ascan to be bg
	bool processBg(int dispModeGS, float &a2, float &a3, uint16_t *mem_data_ptr);
	bool processFrameData(std::string fnameData2, std::string datatype);
	bool processFrameData(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData);
	void setIOPofTiffApi(const UltraOCT::IOP iop);
	void setMemData(uint16_t *mem_data_ptr);
	void setMemBg(uint16_t *mem_bg_ptr);
	void setDataPath(std::string fnameData);
	void setFlatFlag(bool flat);
	bool setBg(uint16_t *mem_bg, uint16_t *mem_data_ptr);
	void setGrayDynamic(int gray);
	void setGammaDynamic(float gamma);
	void setContrastADynamic(int contrastA);
	// re-correct diff
	void CorrectDiff(std::string path);
	void blurDiff(std::vector<int> &diff);
	void WrappercudaDeviceReset();

	//use to grid search a2, a3 will be fix zero
	void gridSearchA2(uint16_t *mem_data_ptr, float& a2, float& a3, std::string fnameData2, std::string tiff_path, std::string datatype, float step = 0.0001);
	double get_img_clarity(std::string path);// get image quality(EVA sharpness)
};

