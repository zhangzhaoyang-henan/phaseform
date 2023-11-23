#pragma once
#include <string>
#include <stdint.h>
class ProcessorCalibration
{
	friend class ProcessorApi;

private:
	int m_batchFrames;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
	int m_batchAscans;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
	int m_prevFreq;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
	bool m_reslice;						// reslice into enface view								
	// Image settings. Same as MATLAB.

	int m_numAscansPerBscan;
	int m_numBScans;
	int m_avgBscan;
	int m_numCameraPixels;
	int m_numBgFrame;

	int m_startPixel;
	int m_endPixel;

	//revised by Tao remove the alpha, add contrastA and contrastB used for adjusting the contrast
	int m_contrastA, m_contrastB;
	float m_alpha;

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	int m_cameraStartPixel;
	int m_cameraEndPixel;
	int m_numUsedPixels;

	int m_grayLevel;
	float m_gamma;
	bool m_interpLinear;

	bool m_dispModeGS;
	int m_dispFrame;
	int m_dispTests;
	float m_dispA2;
	float m_dispA3;

	float m_crop_percent;
	int m_median_filter_size;
	float m_bg_noise, m_bg_mask_correct;

	std::string m_fnamePhase;
	std::string m_fnameBg;
	std::string m_fnameData;

public:
	ProcessorCalibration(void) {};
	~ProcessorCalibration(void) {};

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameBg, std::string fnameData,
		int contrastA, int contrastB,
		float crop_percent, int median_filter_size)
	{
		// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
		m_cameraStartPixel = cameraStartPixel;
		m_cameraEndPixel = cameraEndPixel;
		m_numUsedPixels = numUsedPixels;

		m_batchFrames = batchFrames;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
		m_batchAscans = batchAscans;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
		m_prevFreq = prevFreq;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
		m_reslice = reslice;						// reslice into enface view								
		// Image settings. Same as MATLAB.

		m_numAscansPerBscan = numAscansPerBscan;
		m_numBScans = numBScans;
		m_avgBscan = avgBscan;
		m_numCameraPixels = numCameraPixels;
		m_numBgFrame = numBgFrame;

		m_startPixel = startPixel;
		m_endPixel = endPixel;
		//revised by Tao use fixed alpha:0.5
		m_alpha = 0.5;
		m_contrastA = contrastA;
		m_contrastB = contrastB;

		m_grayLevel = grayLevel;
		m_interpLinear = interpLinear;

		m_dispModeGS = dispModeGS;
		m_dispFrame = dispFrame;
		m_dispTests = dispTests;
		m_dispA2 = dispA2;
		m_dispA3 = dispA3;

		m_fnamePhase = fnamePhase;
		m_fnameBg = fnameBg;
		m_fnameData = fnameData;

		m_crop_percent =  crop_percent;
		m_median_filter_size = median_filter_size;

	}

	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		int contrastA, int contrastB,
		float crop_percent, int median_filter_size)
	{
		// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
		m_cameraStartPixel = cameraStartPixel;
		m_cameraEndPixel = cameraEndPixel;
		m_numUsedPixels = numUsedPixels;

		m_batchFrames = batchFrames;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
		m_batchAscans = batchAscans;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
		m_prevFreq = prevFreq;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
		m_reslice = reslice;						// reslice into en-face view								
		// Image settings. Same as MATLAB.

		m_numAscansPerBscan = numAscansPerBscan;
		m_numBScans = numBScans;
		m_avgBscan = avgBscan;
		m_numCameraPixels = numCameraPixels;
		m_numBgFrame = numBgFrame;

		m_startPixel = startPixel;
		m_endPixel = endPixel;
		//revised by Tao use fixed alpha:0.5
		m_alpha = 0.5;
		m_contrastA = contrastA;
		m_contrastB = contrastB;

		m_grayLevel = grayLevel;
		m_interpLinear = interpLinear;

		m_dispModeGS = dispModeGS;
		m_dispFrame = dispFrame;
		m_dispTests = dispTests;
		m_dispA2 = dispA2;
		m_dispA3 = dispA3;

		m_fnamePhase = fnamePhase;
		this->m_fnameData = fnameData;

		m_crop_percent = crop_percent;
		m_median_filter_size = median_filter_size;
	}
	// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
	void setProcessorCalibration(int batchFrames, int batchAscans, int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans, int avgBscan, int numCameraPixels, int numBgFrame,
		int cameraStartPixel, int cameraEndPixel, int numUsedPixels,
		int startPixel, int endPixel, int grayLevel, float gamma, bool interpLinear,
		bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		int contrastA, int contrastB, float crop_percent, int median_filter_size, 
		float bg_noise, float bg_mask_correct)
	{
		// add by tao @2017 04 06: to crop the raw data being in the range of [cameraStartPixel, cameraEndPixel]
		m_cameraStartPixel = cameraStartPixel;
		m_cameraEndPixel = cameraEndPixel;
		m_numUsedPixels = numUsedPixels;

		m_batchFrames = batchFrames;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
		m_batchAscans = batchAscans;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
		m_prevFreq = prevFreq;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
		m_reslice = reslice;						// reslice into en-face view								
													// Image settings. Same as MATLAB.

		m_numAscansPerBscan = numAscansPerBscan;
		m_numBScans = numBScans;
		m_avgBscan = avgBscan;
		m_numCameraPixels = numCameraPixels;
		m_numBgFrame = numBgFrame;

		m_startPixel = startPixel;
		m_endPixel = endPixel;
		//revised by Tao use fixed alpha:0.5
		m_alpha = 0.5;
		m_contrastA = contrastA;
		m_contrastB = contrastB;

		m_grayLevel = grayLevel;
		this->m_gamma = gamma;
		m_interpLinear = interpLinear;

		m_dispModeGS = dispModeGS;
		m_dispFrame = dispFrame;
		m_dispTests = dispTests;
		m_dispA2 = dispA2;
		m_dispA3 = dispA3;

		m_fnamePhase = fnamePhase;
		this->m_fnameData = fnameData;

		m_crop_percent = crop_percent;
		m_median_filter_size = median_filter_size;
		this->m_bg_noise = bg_noise;
		this->m_bg_mask_correct = bg_mask_correct;
	}

	void resetDispersion(bool dispModeGS, int dispFrame, int dispTests, float dispA2, float dispA3)
	{
		m_dispModeGS = dispModeGS;
		m_dispFrame = dispFrame;
		m_dispTests = dispTests;
		m_dispA2 = dispA2;
		m_dispA3 = dispA3;
	}
};

