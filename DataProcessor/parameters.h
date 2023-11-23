#pragma once

#include <iostream>
#include <fstream>
#include <stdint.h>
#include "TiffHandler.h"
/*
*
* To Do: Make variables private and use getters and setters
*/

class Parameters
{
public:
	// Default constructor is for debug and test cases
	Parameters();

	// This constructor takes input from the GUI through the wrapper DLL.
	Parameters(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans,
		int avgBscan, int numCameraPixels,
		int numUsedPixels, int camStartPixel,
		int camEndPixel, int numBgFrame,
		int startPixel, int endPixel,
		float alpha, float grayLevel,
		int contrastA, int contrastB,
		bool interpLinear, bool dispModeGS,
		int dispFrame, int dispTests,
		float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameBg, std::string fnameData,
		float crop_percent, int median_filter_size
		);
	Parameters(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans,
		int avgBscan, int numCameraPixels,
		int numUsedPixels, int camStartPixel,
		int camEndPixel, int numBgFrame,
		int startPixel, int endPixel,
		float alpha, float grayLevel,
		int contrastA, int contrastB,
		bool interpLinear, bool dispModeGS,
		int dispFrame, int dispTests,
		float dispA2, float dispA3,
		std::string fnamePhase, std::string fnameData,
		float crop_percent, int median_filter_size
		);
	Parameters(int batchFrames, int batchAscans,
		int prevFreq, bool reslice,
		int numAscansPerBscan, int numBScans,
		int avgBscan, int numCameraPixels,
		int numUsedPixels, int camStartPixel,
		int camEndPixel, int numBgFrame,
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


	~Parameters(void);
	// Speed Optimization. These go first since batchFrames needs to be initialized before ascan and bscan values.
	int batchFrames;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
	int batchAscans;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
	int prevFreq;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
	bool reslice;						// reslice into enface view								
	// Image settings. Same as MATLAB.
	int numAscansPerBscan;
	int numBScans;
	int avgBscan;
	int numCameraPixels;
	int numUsedPixels;

	int camStartPixel;
	int camEndPixel;

	int numBgFrame;
	int startPixel;
	int endPixel;
	float alpha;
	int grayLevel;
	float gamma; //[0.05, 5.0]

	int contrastA;
	int contrastB;

	bool interpLinear;					// interpolation mode - linear or cubic spline
	// Dispersion Settings
	bool dispModeGS;					// Gridsearch Mode or Manual Entry Mode
	int dispFrame;						// Frame number to use to calculate A2 & A3
	int dispTests;						// Number of tests to use for gridsearch
	float dispA2;
	float dispA3;
	// Other & File paths
	std::string fnamePhase;
	std::string fnameBg;
	std::string fnameData;

	//ptr of the raw data stored in the memory
	uint16_t *mem_data_ptr, *mem_bg_ptr;

	//info of patient
	UltraOCT::IOP iop;

	//add by Junchao 1/24/2018
	float crop_percent;
	int median_filter_size;

	//add by tao for bg_mask and bg_nosie
	float bg_noise, bg_mask_correct;
	//whether flat the image or not
	bool flat;
};