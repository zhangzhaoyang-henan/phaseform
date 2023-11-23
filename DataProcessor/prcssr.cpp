#pragma once

#include "prcssr.h"
#include <ctime>

using namespace std;

Prcssr::Prcssr(int batchFrames, int batchAscans,
	int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan,
	int numCameraPixels, int numUsedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
	int startPixel, int endPixel,
	float alpha, float grayLevel,
	int contrastA, int contrastB,
	bool interpLinear, bool dispModeGS,
	int dispFrame, int dispTests,
	float dispA2, float dispA3,
	string fnamePhase, string fnameBg, string fnameData,
	float crop_percent, int median_filter_size
	) :
	Parameters(batchFrames, batchAscans,
	prevFreq, reslice,
	numAscansPerBscan, numBScans, avgBscan,
	numCameraPixels, numUsedPixels, camStartPixel, camEndPixel, numBgFrame,
	startPixel, endPixel,
	alpha, grayLevel,
	contrastA, contrastB,
	interpLinear, dispModeGS,
	dispFrame, dispTests,
	dispA2, dispA3,
	fnamePhase, fnameBg, fnameData,
	crop_percent, median_filter_size
	)
{
	p = NULL;
	data = NULL;
	interp = NULL;
	disp = NULL;
	bg = NULL;
	out = NULL;
}

Prcssr::Prcssr(int batchFrames, int batchAscans,
	int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan,
	int numCameraPixels, int numUsedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
	int startPixel, int endPixel,
	float alpha, float grayLevel,
	int contrastA, int contrastB,
	bool interpLinear, bool dispModeGS,
	int dispFrame, int dispTests,
	float dispA2, float dispA3,
	string fnamePhase, string fnameData,
	float crop_percent, int median_filter_size
	) :
	Parameters(batchFrames, batchAscans,
	prevFreq, reslice,
	numAscansPerBscan, numBScans, avgBscan,
	numCameraPixels, numUsedPixels, camStartPixel, camEndPixel, numBgFrame,
	startPixel, endPixel,
	alpha, grayLevel,
	contrastA, contrastB,
	interpLinear, dispModeGS,
	dispFrame, dispTests,
	dispA2, dispA3,
	fnamePhase, fnameData,
	crop_percent, median_filter_size
	)
{
	p = NULL;
	data = NULL;
	interp = NULL;
	disp = NULL;
	bg = NULL;
	out = NULL;
}
Prcssr::Prcssr(int batchFrames, int batchAscans,
	int prevFreq, bool reslice,
	int numAscansPerBscan, int numBScans, int avgBscan,
	int numCameraPixels, int numUsedPixels, int camStartPixel, int camEndPixel, int numBgFrame,
	int startPixel, int endPixel,
	float alpha, float grayLevel, float gamma,
	int contrastA, int contrastB,
	bool interpLinear, bool dispModeGS,
	int dispFrame, int dispTests,
	float dispA2, float dispA3,
	string fnamePhase, string fnameData,
	float crop_percent, int median_filter_size,
	float bg_noise, float bg_mask_correct
) :
	Parameters(batchFrames, batchAscans,
		prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan,
		numCameraPixels, numUsedPixels, camStartPixel, camEndPixel, numBgFrame,
		startPixel, endPixel,
		alpha, grayLevel,gamma,
		contrastA, contrastB,
		interpLinear, dispModeGS,
		dispFrame, dispTests,
		dispA2, dispA3,
		fnamePhase, fnameData,
		crop_percent, median_filter_size,
		bg_noise, bg_mask_correct
	)
{
	p = NULL;
	data = NULL;
	interp = NULL;
	disp = NULL;
	bg = NULL;
	out = NULL;
}
Prcssr::~Prcssr()
{
	std::cout << __FUNCTION__ << "\t\t	go in ~Prcssr()" << std::endl;
	if (bg) { delete bg; bg = NULL; }
	if (disp) { delete disp; disp = NULL; }
	if (interp) { delete interp; interp = NULL; }
	if (data) { delete data; data = NULL; }
	if (p) { delete p; p = NULL; }
	std::cout << __FUNCTION__ << "\t\t	go out ~Prcssr()" << std::endl;
}

void Prcssr::runParam()
{
	//cudaDeviceReset();

	p = new Parameters(batchFrames, batchAscans,
		prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan,
		numCameraPixels, numUsedPixels, camStartPixel, camEndPixel, numBgFrame,
		startPixel, endPixel,
		alpha, grayLevel,gamma, 
		contrastA, contrastB,
		interpLinear, dispModeGS,
		dispFrame, dispTests,
		dispA2, dispA3,
		fnamePhase, fnameData,
		crop_percent, median_filter_size,
		bg_noise, bg_mask_correct
		);
}

void Prcssr::runData()
{
	data = new Data(*p);
}

void Prcssr::runInterp()
{
	interp = new Interpolation(*p, *data);
	interp->init();
}

void Prcssr::runDisp()
{
	disp = new Dispersion(*p, *data, *interp);
	disp->process();
}

void Prcssr::runDisp(int dispModeGS, float &a2, float &a3)
{
	disp = new Dispersion(*p, *data, *interp);
	disp->process(dispModeGS, a2, a3);
}

void Prcssr::runDisp(int dispModeGS, float &a2, float &a3, uint16_t *mem_init_data_ptr)
{
	p->mem_data_ptr = mem_init_data_ptr;
	disp = new Dispersion(*p, *data, *interp);
	disp->process(dispModeGS, a2, a3);
}

void Prcssr::runBg()
{
	bg = new Background(*p, *data, *interp, *disp);
	bg->process();
}

void Prcssr::runBg(uint16_t *mem_bg_ptr)
{
	p->mem_bg_ptr = mem_bg_ptr;
	bg = new Background(*p, *data, *interp, *disp);
	bg->process();
}

void Prcssr::runOutput()
{
	out = new Output(*p, *data, *interp, *disp, *bg);
	//out->process();

	if (out) { delete out; out = NULL; }
}

void Prcssr::runOutput(string str)
{
	p->fnameData = str;
	out = new Output(*p, *data, *interp, *disp, *bg);
	//out->process();

	if (out) { delete out; out = NULL; }
}

void Prcssr::runOutput(string str, std::string datatype)
{
	std::vector<uint8_t> bmpRawData;
	std::vector<float> tmpvec;
	p->fnameData = str;
	out = new Output(*p, *data, *interp, *disp, *bg);
	out->process(datatype, bmpRawData, tmpvec);

	if (out) { delete out; out = NULL; }
}

void Prcssr::runOutput(uint16_t *mem_data_ptr, std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData)
{
	std::clock_t start = clock();
	p->mem_data_ptr = mem_data_ptr;
	out = new Output(*p, *data, *interp, *disp, *bg);
	std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for output constructor:	" << clock() - start << "ms" << std::endl;
	out->process(datatype, bmpRawData, previewFFTrawData);
	std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for output process:	" << clock() - start << "ms" << std::endl;
	start = clock();
	if (out) { delete out; out = NULL; }
	std::cout << __FUNCTION__ << "\t" << __TIMESTAMP__ << "\t" << "time for output delete:	" << clock() - start << "ms" << std::endl;
}

void Prcssr::setIOPofTiff(const UltraOCT::IOP iop)
{
	this->p->iop = iop;
}

void Prcssr::setMemData(uint16_t *mem_data_ptr)
{
	this->p->mem_data_ptr = mem_data_ptr;
}

void Prcssr::setMemBg(uint16_t *mem_bg_ptr)
{
	this->p->mem_bg_ptr = mem_bg_ptr;
}

void Prcssr::setGrayDynamic(int gray)
{
	this->p->grayLevel = gray;
}

void Prcssr::setGammaDynamic(float gamma)
{
	this->p->gamma = gamma;
}

void Prcssr::setDataPath(std::string fnamePath)
{
	this->p->fnameData = fnamePath;
}

void Prcssr::setFlatFlag(bool flat)
{
	this->p->flat = flat;
}

void Prcssr::setA2AndA3(float a2, float a3)
{
	this->p->dispA2 = a2;
	this->p->dispA3 = a3;
}

void Prcssr::freeBg(string str)
{
	if (bg) { delete bg; bg = NULL; }
	p->fnameBg = str;
}

void Prcssr::releaseDisp()
{
	if (this->disp)
	{
		delete this->disp;
		this->disp = nullptr;
	}
}

void Prcssr::releaseBg()
{
	if (this->bg)
	{
		delete this->bg;
		this->bg = nullptr;
	}
}