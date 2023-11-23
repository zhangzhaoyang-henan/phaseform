#pragma once

#include "output.cuh"

class Prcssr :public Parameters
{
public:
	Prcssr(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan,
		int numCameraPixels, int numusedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
		int startPixel, int endPixel,
		float alpha, float grayLevel,
		int contrastA, int contrastB,
		bool interpLinear, bool dispModeGS,
		int dispFrame, int dispTests,
		float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameBg, std::string fnameData,
		float crop_percent, int median_filter_size
		);
	Prcssr(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan,
		int numCameraPixels, int numusedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
		int startPixel, int endPixel,
		float alpha, float grayLevel,
		int contrastA, int contrastB,
		bool interpLinear, bool dispModeGS,
		int dispFrame, int dispTests,
		float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		float crop_percent, int median_filter_size
		);
	Prcssr(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan,
		int numCameraPixels, int numusedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
		int startPixel, int endPixel,
		float alpha, float grayLevel, float gamma,
		int contrastA, int contrastB,
		bool interpLinear, bool dispModeGS,
		int dispFrame, int dispTests,
		float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		float crop_percent, int median_filter_size,
		float bg_noise, float bg_mask_correct
	);

	~Prcssr(void);

	void runParam();
	void runData();
	void runInterp();
	void runDisp();
	void runDisp(int dispModeGS, float &a2, float &a3);
	void runDisp(int dispModeGS, float &a2, float &a3, uint16_t *mem_init_data_ptr);

	void runBg();
	void runBg(uint16_t * mem_bg_ptr);
	void runOutput();					// The version of output that takes a string runs when batching is used,
	void runOutput(std::string str);	// str contains the path to the next data file.
	void runOutput(std::string str, std::string datatype);	// str contains the path to the next data file.
	void runOutput(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData);

	void setIOPofTiff(const UltraOCT::IOP iop);
	void setMemData(uint16_t* mem_data_ptr);
	void setMemBg(uint16_t *mem_bg_ptr);
	void setGrayDynamic(int gray);
	void setGammaDynamic(float gamma);
	//dynamically set the path to store the processed data
	void setDataPath(std::string fnamePath);
	void setFlatFlag(bool flat);
	//dynamically set a2  and a3
	void setA2AndA3(float a2, float a3);
	// When batching directories, we need to recalculate the background for each folder. To do this, the background
	// class needs to be freed and recreated. This function handles that, passing in the path to the new background file.	
	void freeBg(std::string str);

	//releaseDisp for redo
	void releaseDisp();
	//releaseBg for redo
	void releaseBg();

private:
	Parameters *p;
	Data *data;
	Interpolation *interp;
	Dispersion *disp;
	Background *bg;
	Output *out;
};
