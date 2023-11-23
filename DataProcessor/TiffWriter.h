#ifndef _TIFFWRITER_H_
#define _TIFFWRITER_H_
#include <string>
#include <Magick++.h>


//added by tao, 
namespace MagickTiff
{
	class TiffWriter
	{
	public:
		TiffWriter(std::string filename);
		//TiffWriter(const TiffWriter&) = delete;
		//TiffWriter& operator=(const TiffWriter&) = delete;
		~TiffWriter();

		void write(const unsigned char* buffer, int width, int height);
		std::vector<Magick::Image>::iterator begin(){ return this->imageList.begin(); }
		std::vector<Magick::Image>::iterator end(){ return this->imageList.end(); }
		std::string get_filename(){ return this->filename; }
		void Dealloc_imageList();
	private:
		std::vector<Magick::Image> imageList;
		std::string filename;
	};

	TiffWriter::TiffWriter(std::string filename) : filename(filename) 
	{
		Magick::InitializeMagick(NULL);
	}

	// for example for a 8 bit gray image buffer
	void TiffWriter::write(const unsigned char* buffer, int width, int height)
	{
		Magick::Blob gray8Blob(buffer, width * height);
		//Magick::Geometry size(width, height);
		Magick::Geometry size(height, width);
		Magick::Image gray8Image(gray8Blob, size, 8, "GRAY");
		imageList.push_back(gray8Image);
	}
	void TiffWriter::Dealloc_imageList()
	{
		imageList.swap(std::vector<Magick::Image>());
	}
	TiffWriter::~TiffWriter()
	{
		;//Magick::writeImages(imageList.begin(), imageList.end(), filename);
	}
}
#endif