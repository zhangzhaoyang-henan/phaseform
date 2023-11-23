#include "TiffHandler.h"
using namespace UltraOCT;


TiffHandler::TiffHandler()
{
	/*this->iop.gender = 0;
	this->iop.age = 26;
	this->iop.hpv = 1;
	this->iop.tct = 1;*/
	this->iop.xpixel_size = 1.31;
	this->iop.ypixel_size = 1.52;
	this->setPDE();
}

TiffHandler::TiffHandler(IOP inIop)
{
	//init the patient info
	//this->iop = inIop;
	this->SetTiffIOP(inIop);
	this->setPDE();

}


TiffHandler::~TiffHandler()
{
	/*if (this->fp)
	{
		fclose(fp);
	}*/
}

void TiffHandler::setPDE()
{
	//init the ifh for tiff
	ifh.Byte_order = 0x4949; //"II", 小字节在先
	ifh.Version = 0x2a; // dec is 42 to tell this is a tiff file
	ifh.OffsetToFirstFID = 0;

	//init the tag used in this tiff handler
	pde.wDECount = 14;
	pde.pde = new DE[pde.wDECount];

	//please refer to url: https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html to get the meaning for each tag
	//tag 254
	pde.pde[0].tag = 254;
	pde.pde[0].type = 4;
	pde.pde[0].length = 1;
	pde.pde[0].valueOffset = 2;

	//tag: 256
	//name: ImageWidth
	pde.pde[1].tag = 256;
	pde.pde[1].type = 3;
	pde.pde[1].length = 1;
	pde.pde[1].valueOffset = 0;

	//tag:	257
	//name:	ImageLength/rows
	pde.pde[2].tag = 257;
	pde.pde[2].type = 3;
	pde.pde[2].length = 1;
	pde.pde[2].valueOffset = 0;

	//tag:	258
	//name:	BitsPerSample
	pde.pde[3].tag = 258;
	pde.pde[3].type = 3;
	pde.pde[3].length = 1;
	pde.pde[3].valueOffset = 8;

	//tag 259
	pde.pde[4].tag = 259;
	pde.pde[4].type = 3;
	pde.pde[4].length = 1;
	pde.pde[4].valueOffset = 1;

	//tag 262
	pde.pde[5].tag = 262;
	pde.pde[5].type = 3;
	pde.pde[5].length = 1;
	pde.pde[5].valueOffset = 1;

	//tag 266
	pde.pde[6].tag = 266;
	pde.pde[6].type = 3;
	pde.pde[6].length = 1;
	pde.pde[6].valueOffset = 1;

	//tag:	273
	//name: page offset compare to the beginning of tiff file
	//point to raw data of each page
	pde.pde[7].tag = 273;
	pde.pde[7].type = 4;
	pde.pde[7].length = 1;
	pde.pde[7].valueOffset = 0;

	//tag 274
	pde.pde[8].tag = 274;
	pde.pde[8].type = 3;
	pde.pde[8].length = 1;
	pde.pde[8].valueOffset = 1;

	//tag 277
	pde.pde[9].tag = 277;
	pde.pde[9].type = 3;
	pde.pde[9].length = 1;
	pde.pde[9].valueOffset = 1;

	//tag:	278
	//name: row/height
	pde.pde[10].tag = 278;
	pde.pde[10].type = 3;
	pde.pde[10].length = 1;
	pde.pde[10].valueOffset = 0;

	//tag:	279
	//name: columns/width
	pde.pde[11].tag = 279;
	pde.pde[11].type = 4;
	pde.pde[11].length = 1;
	pde.pde[11].valueOffset = 0;

	//tag 284
	pde.pde[12].tag = 284;
	pde.pde[12].type = 3;
	pde.pde[12].length = 1;
	pde.pde[12].valueOffset = 1;

	//tag 297
	//page number: total/current
	pde.pde[13].tag = 297;
	pde.pde[13].type = 3;
	pde.pde[13].length = 2;
	pde.pde[13].valueOffset = 0;
	//1310720, to binary is: 10100, 0000000000000000, stand for to short integers: 20, 0,
	//which means the entire tiff has 20 pages, and the current page number is 0 
}

bool TiffHandler::OpenTiffFile(const char* path)
{
	fopen_s(&(this->fp), path, "wb+");
	if (this->fp == NULL)
	{
		std::cout<< __TIME__ <<__FUNCTION__ << " cannot open file for writing" << std::endl;
		return false;
	}
	return true;
}

void TiffHandler::CloseTiffFile()
{
	if (this->fp != NULL)
	{
		fclose(this->fp);
	}
	if (this->pde.pde != nullptr)
	{
		delete[]this->pde.pde;
	}
}

//set and write tiff head to file
bool TiffHandler::SetIFDandTiffParams(Size& local_img_size, int bitsPersample, int tpage)
{
	this->img_size = local_img_size;
	if (this->fp == NULL)
	{
		std::cout << __TIME__ << __FUNCTION__ << "tiff file is not exist!" << std::endl;
		return false;
	}
	this->ifh.OffsetToFirstFID = sizeof(IFH) + sizeof(IOP) + 
		this->img_size.width*this->img_size.height;
	this->curOffset = this->ifh.OffsetToFirstFID;
	if (sizeof(IFH) != fwrite(&(this->ifh), 1, sizeof(IFH), fp))
	{
		std::cout << __TIME__ << __FUNCTION__ << "cannot write ifh to file!" << std::endl;
		return false;
	}

	if (sizeof(IOP) != fwrite(&(this->iop), 1, sizeof(iop), fp))
	{
		std::cout << __TIME__ << __FUNCTION__ << "cannot write iop to file!" << std::endl;
		return false;
	}
	pde.pde[1].valueOffset = this->img_size.width;
	pde.pde[2].valueOffset = this->img_size.height;
	pde.pde[3].valueOffset = bitsPersample;
	pde.pde[7].valueOffset = sizeof(IFH) + sizeof(IOP);// change for every page
	pde.pde[10].valueOffset = this->img_size.height;
	pde.pde[11].valueOffset = this->img_size.width * this->img_size.height;
	this->total_page = tpage;
	return true;
}

//wrire the given page to file
bool TiffHandler::writePage(const unsigned char* data, int page_num)
{
	/*std::cout << "page:	" << page_num << "\t" << "height:	" << this->img_size.height << "\t"
		<< "width:	" << this->img_size.width << std::endl;*/

	this->pde.pde[13].valueOffset = (this->total_page << 16) | page_num;
	unsigned long long data_begin_point = sizeof(IFH) + sizeof(IOP) + 
		page_num * (DEBYTESNUM * this->pde.wDECount + DECOUNTBYTES + NEXTIFDBYTES + 
			this->img_size.height*this->img_size.width);
	pde.pde[7].valueOffset = data_begin_point;
	int j = 0;
	//go the current DE
	if (0 != fseek(this->fp, this->curOffset, SEEK_SET))
	{
		std::cout << __DATE__ << __FUNCTION__ << "cannot seek to:	" << this->curOffset << std::endl;
		return false;
	}

	//write DE count to file
	if (2 != fwrite(&this->pde.wDECount, 1, sizeof(unsigned short), this->fp))
	{
		std::cout << __DATE__ << __FUNCTION__ << "cannot write DECount to file!" << std::endl;
		return false;
	}

	//write DEs to file
	if (sizeof(DE)*this->pde.wDECount != fwrite(this->pde.pde, 1, sizeof(DE)*this->pde.wDECount, this->fp))
	{
		std::cout << __DATE__ << __FUNCTION__ << "cannot write DEs to file!" << std::endl;
		return false;
	}
	
	this->curOffset += DEBYTESNUM * this->pde.wDECount + DECOUNTBYTES + NEXTIFDBYTES +
		this->img_size.height*this->img_size.width;
	if (page_num != this->total_page-1)
	{
		if (4 != fwrite(&this->curOffset, 1, sizeof(unsigned int), this->fp))
		{
			std::cout << __DATE__ << __FUNCTION__ << "cannot write DECount to file!" << std::endl;
			return false;
		}
	}
	else
	{
		unsigned int tmp = 0;
		if (4 != fwrite(&tmp, 1, sizeof(unsigned int), this->fp))
		{
			std::cout << __DATE__ << __FUNCTION__ << "cannot write DECount to file!" << std::endl;
			return false;
		}
	}
	


	//seek to the beginning of the raw data
	if (0 != fseek(this->fp, data_begin_point, SEEK_SET))
	{
		std::cout << __DATE__ << __FUNCTION__ << "cannot seek to:	" << data_begin_point << std::endl;
		return false;
	}
	
	if (this->img_size.width*this->img_size.height != fwrite((unsigned char*)(data), sizeof(unsigned char),
		this->img_size.width*this->img_size.height, fp))
	{
		std::cout << __DATE__ << __FUNCTION__ << "cannot write image to file!" << std::endl;
		return false;
	}

	return true;
}

bool TiffHandler::SetTiffIOP(const IOP& virIop)
{
	this->iop = virIop;
	return true;
}