#pragma once
#include <cstdio> 
#include <cstring>
#include <iostream>
#define DEBYTESNUM 12
#define DECOUNTBYTES 2
#define NEXTIFDBYTES 4
namespace UltraOCT
{
	//head of the entire multipage file
	typedef struct
	{
		unsigned short Byte_order;//  
		unsigned short Version;// check if the file is TIFF-type file
		unsigned   int OffsetToFirstFID;//offset compare to the begining of file  
										// pointer to DE count in IFD
	}IFH;

	//tag struct to describe each attribute of each frame, e.g.,height, width....
	typedef struct
	{
		unsigned short tag;//num of the tag, 254, 255,....
		unsigned short type;//type of the value
		unsigned int length;//count of value
		unsigned int valueOffset;//value or the pointer to the value if the value is longer than 4 bits
	}DE;
	//size of each image
	typedef struct
	{
		int width;
		int height;
	}Size;
	//image data
	typedef struct
	{
		int *data;
	}DATA;
	//struct to hold the count of DE and the point to the array of DE
	typedef struct
	{
		DE *pde;
		int wDECount;
	}PDE;

	// struct to store the info of patient, which will be put right after the IFH
	//typedef struct
	//{
	//	unsigned short gender; //0 for male, 1 for female
	//	unsigned short age;
	//	int hpv; //hpv test
	//	int tct; //tct test
	//	float xpixel_size; // pixel size of x[um]
	//	float ypixel_size; // pixel size of y[um]
	//}IOP;
	struct IOP
	{
		//unsigned short gender; //0 for male, 1 for female
		//unsigned short age;
		//int hpv; //hpv test
		//int tct; //tct test
		float xpixel_size; // pixel size of x[um]
		float ypixel_size; // pixel size of y[um]
		IOP()
		{
			/*gender = 0;
			age = 0;
			hpv = 0;
			tct = 0;*/
			xpixel_size = 1.0;
			xpixel_size = 1.0;
		}
	};

	//tiff handler
	class TiffHandler
	{
	public:
		TiffHandler();
		TiffHandler(IOP inIop);
		~TiffHandler();

		bool OpenTiffFile(const char* path);
		void CloseTiffFile();
		bool SetIFDandTiffParams(Size& img_size, int bitsPersample, int tpage);
		bool writePage(const unsigned char* data, int page_num);
		//set info to be writen right after the IFH
		bool SetTiffIOP(const IOP& virIop);
		void setPDE();
	private:
		PDE pde;
		IOP iop;
		IFH ifh;
		Size img_size;
		FILE *fp;
		int total_page;
		unsigned long long curOffset;
	};
}

