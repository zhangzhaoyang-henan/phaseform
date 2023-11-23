#include <iostream>
#include <string>
#include <fstream>
#include "ProcessorCalibration.h"
#include "ProcessorApi.h"
#include <time.h>
#include <iomanip>

#if 1
using namespace std;

int batchFrames = 1;					// batch frames. Lowers preview frequency. Lower numbers are fine ~2.
int batchAscans = 10;					// batch A-scans to increase occupancy on GPU. 10 or 20 gets decent results.
int prevFreq = 40;						// 1 means show every frame, 2 means show every other frame, 4 means show 1 in 4 frames, etc. Higher is better, has an decent impact on speed.
bool reslice = false;					// reslice into enface view								
// Image settings. Same as MATLAB.

int numAscansPerBscan = batchFrames * 1200;
int numBScans = 10 / batchFrames;
int avgBscan = 1;
int numCameraPixels = 2048;


int cameraStartPixel = 1;
int cameraEndPixel = 2048;
int numUsedPixels = cameraEndPixel - cameraStartPixel + 1;

int numBgFrame = 0;
int startPixel = 1;
int endPixel = 850;

int contrastA = 400;
int contrastB = 0;
float alpha = 0.5;


int grayLevel = 65;
bool interpLinear = true;

bool dispModeGS = false;
int dispFrame = 1;
int dispTests = 100;
float dispA2 = 0.0000125;
float dispA3 = 0;
float gamma = 1.07;
float crop_percent = 0.00;
int median_filter_size = 1;


bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

void readBin2Vec(std::vector<uint16_t>& raw_data, std::string fname)
{

	uint16_t i;
	int count = numBScans * numAscansPerBscan *  numCameraPixels; //10*1200*2048
	ifstream infile(fname, ios::binary);
	if (!infile)
	{
		cerr << "Can't open file " << fname << " for reading" << endl; exit(EXIT_FAILURE);
	}
	cout << "Opened binary file " << fname << " for reading." << endl;
	while (infile.read(reinterpret_cast<char *>(&i), sizeof(i)) && count-->0)
	{
		raw_data.push_back(i);
	}
	infile.close();
}

int main(int argc, char **argv)
{
	std::string fnamePhase = "F:\\code\\phaseform\\Data\\gongkong.txt";
	std::string fnameBg;
	std::string fnameData;
	std::string fnameData2;
	if (argc == 1)
	{
		//fnameBg = "..\\data\\fundusDataSet.bin";
		fnameData = "F:\\code\\phaseform\\Data\\P01\\M0008_2021_P0000016_circle_3.0x3.1_C2_S2.bin";
		//fnameData2 = ".\\MODEL_M00002_2018_P0000044_S002.bin";
	}
	else if (argc == 4)
	{
		fnamePhase = string(argv[1]);
		fnameBg = string(argv[2]);
		fnameData = string(argv[3]);
		fnameData2 = string(argv[3]);
	}
	else if (argc == 5)
	{
		fnamePhase = string(argv[1]);
		fnameBg = string(argv[2]);
		fnameData = string(argv[3]);
		fnameData2 = string(argv[4]);
	}
	else
	{
		std::cerr << "argument error" << std::endl;
		exit(0);
	}
	//std::string prefix = "D:/taoxu/045zhaoquan_P0000115/MODEL_M00002_2018_P0000115_S";
	std::vector<uint16_t> mem_bg, mem_data;
	
	//readBin2Vec(mem_data, fnameData);
	//将参数写进ProcessorApi 然后调用ProcessorCalibration进行再次赋值，最后将赋值的参数再次写进ProcessorApi调用
	ProcessorApi* prcsrApi = new ProcessorApi();
	prcsrApi->setProcessorCalibration(batchFrames, batchAscans, prevFreq, reslice,
		numAscansPerBscan, numBScans, avgBscan, numCameraPixels, numBgFrame,
		cameraStartPixel, cameraEndPixel, numUsedPixels,
		startPixel, endPixel, grayLevel, gamma, interpLinear,
		dispModeGS, dispFrame, dispTests, dispA2, dispA3,
		fnamePhase, fnameData,
		contrastA, contrastB, crop_percent, median_filter_size,
		30, 0.2);
	readBin2Vec(mem_data, fnameData);
	dispModeGS = false;
	for (size_t i = 1; i <= 1; ++i)
	{
		std::vector<uint8_t> bmpRawData;
		std::vector<float> tmpvec;
		
		clock_t start = clock();
		prcsrApi->processBg(dispModeGS, dispA2, dispA3, &mem_data[0]);
		std::cout << "		process bg time:" << clock() - start << std::endl;

		prcsrApi->setFlatFlag(false); //设置拉平

		start = clock();
		prcsrApi->processFrameData(&mem_data[0], "acquire", bmpRawData, tmpvec);
		std::cout << "		process time:" << clock() - start << std::endl;

		//system("pause");
		
		//mem_data.swap(std::vector<uint16_t>());

	}
	delete prcsrApi;
}
#endif

