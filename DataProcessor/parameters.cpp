#include "parameters.h"

using namespace std;

// Debug or Test Case Constructor
Parameters::Parameters()
{
	cout << "Initialization" << endl;

	int selection = 4;					// select which test case to run. 1 is 2048 pixels, 2 is 1024 pixels.

	if (selection == 1)
	{
		batchFrames = 50;
		batchAscans = 40;
		prevFreq = 40;
		reslice = false;

		numAscansPerBscan = batchFrames * 800;
		numBScans = 800 / batchFrames;
		numCameraPixels = 1024;
		numUsedPixels = 1024;
		numBgFrame = 3;
		startPixel = 15;
		endPixel = 480;
		alpha = 0.5;
		grayLevel = 30;
		interpLinear = true;

		dispModeGS = true;
		dispFrame = 100;
		dispTests = 100;
		dispA2 = 0.00909076;
		dispA3 = 2.44613e-7;

		crop_percent = 0.1;
		median_filter_size = 5;

		fnamePhase = "E:\\Data\\gpu_test_data\\calibration_files\\1024_calibration_file.txt";
		fnameBg = "E:\\Data\\2015-05-29_SHR_e11_5_mouse_embryo_heart\\SHR_e11_5_mouse_embryo_OD_U-3D_ 9x 9_R00.bin";
		fnameData = "E:\\Data\\2015-05-29_SHR_e11_5_mouse_embryo_heart\\SHR_e11_5_mouse_embryo_OD_U-3D_ 9x 9_R01.bin";
	}
	else if (selection == 2)
	{
		batchFrames = 4;
		batchAscans = 10;
		prevFreq = 10;
		reslice = false;

		numAscansPerBscan = batchFrames * 800;
		numBScans = 800 / batchFrames;
		numCameraPixels = 1024;
		numUsedPixels = 1000;
		numBgFrame = 2;
		startPixel = 6;
		endPixel = 480;
		alpha = 0.5;
		grayLevel = 25;
		interpLinear = true;

		dispModeGS = false;
		dispFrame = 400;
		dispTests = 100;
		dispA2 = 0; //0.00909076;
		dispA3 = 0; //2.44613e-7;

		crop_percent = 0.1;
		median_filter_size = 5;

		fnamePhase = "C:\\Users\\andre\\Desktop\\OCT_dataset\\1024_calibration_file.txt";
		fnameBg = "C:\\Users\\andre\\Desktop\\OCT_dataset\\2015-03-18_SHR_mouse_embryo_heart_3disoc_4.5h\\SHR_mouse_embryo_heart_3disoc_4.5h_OD_U-3D_16x16_R00.bin";
		fnameData = "C:\\Users\\andre\\Desktop\\OCT_dataset\\2015-03-18_SHR_mouse_embryo_heart_3disoc_4.5h\\SHR_mouse_embryo_heart_3disoc_4.5h_OD_U-3D_16x16_R03.bin";
	}
	else if (selection == 3)
	{
		batchFrames = 10;
		batchAscans = 2;
		prevFreq = 2;
		reslice = false;

		numAscansPerBscan = batchFrames * 800;
		numBScans = 800 / batchFrames;
		avgBscan = 1;

		numCameraPixels = 1024;
		numUsedPixels = 1000;
		numBgFrame = 2;
		startPixel = 10;
		endPixel = 480;
		alpha = 0.5;
		grayLevel = 25;
		interpLinear = true;

		dispModeGS = false;
		dispFrame = 400;
		dispTests = 100;
		dispA2 = 0.00909076; //0.00909076;
		dispA3 = 2.44613e-7; //2.44613e-7;

		crop_percent = 0.1;
		median_filter_size = 5;

		fnamePhase = "C:\\Users\\andre\\Desktop\\OCT_dataset\\1024_calibration_file.txt";
		fnameBg = "C:\\Users\\andre\\Desktop\\OCT_dataset\\beef\\SHR_beef_3disco_OD_U-3D_33x33_R00.bin";
		fnameData = "C:\\Users\\andre\\Desktop\\OCT_dataset\\beef\\SHR_beef_3disco_OD_U-3D_33x33_R03.bin";
	}
	else if (selection == 4)
	{
		batchFrames = 1;
		batchAscans = 40;
		prevFreq = 40;
		reslice = false;

		numAscansPerBscan = batchFrames * 600;
		numBScans = 600 / batchFrames;
		/*numAscansPerBscan = 600;
		numBScans = 600;*/
		numCameraPixels = 4096;

		camStartPixel = 1;
		camEndPixel = 4000;
		numUsedPixels = camEndPixel - camStartPixel + 1;
		numBgFrame = 2;
		startPixel = 200;
		endPixel = 1100;
		alpha = 0.5;
		grayLevel = 40;
		contrastA = 384;
		contrastB = -64;
		interpLinear = true;

		dispModeGS = true;  //true means using grid search
		dispFrame = 400;
		dispTests = 100;
		dispA2 = 0.0878787;
		dispA3 = -2.09615e-006;

		crop_percent = 0.1;
		median_filter_size = 5;

		fnamePhase = "Z:/TaoXu/ZZU_Data/PhaseCalibration_ZZU_011916.txt";
		fnameBg = "C:/Users/WhuXu/Desktop/042517/2017-04-25_SHR_C442S1M1/SHR_C442S1M1_OD_U-3D_40x40_R00.bin";
		fnameData = "C:/Users/WhuXu/Desktop/042517/2017-04-25_SHR_C442S1M1/SHR_C442S1M1_OD_U-3D_40x40_R03.bin";
	}
	else if (selection == 5)
	{
		batchFrames = 256;
		batchAscans = 32;
		prevFreq = 40;
		reslice = false;

		numAscansPerBscan = batchFrames * 128;
		numBScans = 4096 / batchFrames;
		/*numAscansPerBscan = 600;
		numBScans = 600;*/
		numCameraPixels = 2048;
		numUsedPixels = 2048;
		numBgFrame = 2;
		startPixel = 10;
		endPixel = 600;
		alpha = 0.5;
		grayLevel = 40;
		interpLinear = true;

		dispModeGS = false;  //true means using grid search
		dispFrame = 400;
		dispTests = 100;
		dispA2 = 0.081818;
		dispA3 = -2.83597e-006;

		crop_percent = 0.1;
		median_filter_size = 5;

		fnamePhase = "D:/Tao Xu/Fly data/e2V_PhaseCalibration_rod10x_122314.txt";
		fnameBg = "D:/Tao Xu/Fly data/2016-09-20_SHR_S02_53748-latepupa-25degree-3.5-4-4.5Hz-1ms-100%/SHR_S02_53748-latepupa-25degree-3.5-4-4.5Hz-1ms-100%_OD_U-3D_ 4x 0_R00.bin";
		fnameData = "D:/Tao Xu/Fly data/2016-09-20_SHR_S02_53748-latepupa-25degree-3.5-4-4.5Hz-1ms-100%/SHR_S02_53748-latepupa-25degree-3.5-4-4.5Hz-1ms-100%_OD_U-3D_ 4x 0_R01.bin";
	}
	cout << "	- Settings loaded" << endl;
}

// Constructor used by the wrapper.
Parameters::Parameters(int batchFrames, int batchAscans,
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
	batchFrames(batchFrames), batchAscans(batchAscans),
	prevFreq(prevFreq), reslice(reslice),
	numAscansPerBscan(numAscansPerBscan), numBScans(numBScans), avgBscan(avgBscan),
	numCameraPixels(numCameraPixels), numUsedPixels(numUsedPixels), camStartPixel(camStartPixel), camEndPixel(camEndPixel), numBgFrame(numBgFrame),
	startPixel(startPixel), endPixel(endPixel),
	alpha(alpha), grayLevel(grayLevel),
	contrastA(contrastA), contrastB(contrastB),
	interpLinear(interpLinear), dispModeGS(dispModeGS),
	dispFrame(dispFrame), dispTests(dispTests),
	dispA2(dispA2), dispA3(dispA3),
	fnamePhase(fnamePhase), fnameBg(fnameBg), fnameData(fnameData),
	crop_percent(crop_percent), median_filter_size(median_filter_size)
{
}

// Constructor used by the wrapper.
Parameters::Parameters(int batchFrames, int batchAscans,
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
	batchFrames(batchFrames), batchAscans(batchAscans),
	prevFreq(prevFreq), reslice(reslice),
	numAscansPerBscan(numAscansPerBscan), numBScans(numBScans), avgBscan(avgBscan),
	numCameraPixels(numCameraPixels), numUsedPixels(numUsedPixels), camStartPixel(camStartPixel), camEndPixel(camEndPixel), numBgFrame(numBgFrame),
	startPixel(startPixel), endPixel(endPixel),
	alpha(alpha), grayLevel(grayLevel),
	contrastA(contrastA), contrastB(contrastB),
	interpLinear(interpLinear), dispModeGS(dispModeGS),
	dispFrame(dispFrame), dispTests(dispTests),
	dispA2(dispA2), dispA3(dispA3),
	fnamePhase(fnamePhase), fnameData(fnameData),
	crop_percent(crop_percent), median_filter_size(median_filter_size)
{
}
Parameters::Parameters(int batchFrames, int batchAscans,
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
	batchFrames(batchFrames), batchAscans(batchAscans),
	prevFreq(prevFreq), reslice(reslice),
	numAscansPerBscan(numAscansPerBscan), numBScans(numBScans), avgBscan(avgBscan),
	numCameraPixels(numCameraPixels), numUsedPixels(numUsedPixels), camStartPixel(camStartPixel), camEndPixel(camEndPixel), numBgFrame(numBgFrame),
	startPixel(startPixel), endPixel(endPixel),
	alpha(alpha), grayLevel(grayLevel), gamma(gamma),
	contrastA(contrastA), contrastB(contrastB),
	interpLinear(interpLinear), dispModeGS(dispModeGS),
	dispFrame(dispFrame), dispTests(dispTests),
	dispA2(dispA2), dispA3(dispA3),
	fnamePhase(fnamePhase), fnameData(fnameData),
	crop_percent(crop_percent), median_filter_size(median_filter_size),
	bg_noise(bg_noise), bg_mask_correct(bg_mask_correct)
{
	this->flat = true;
}

Parameters::~Parameters()
{
}