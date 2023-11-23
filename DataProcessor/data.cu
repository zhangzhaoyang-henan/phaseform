#include "data.cuh"

using namespace std;

Data::Data(Parameters &p) : Helper(p)
{
	d_gauss_win = NULL;
	genGaussWin();
}

Data::~Data()
{
	if (d_gauss_win) gpuErrchk(cudaFree(d_gauss_win));
}

void Data::genGaussWin()  //¸ßË¹´°¿Ú
{
	float temp;
	float *gauss_win = new float[width_2x];
	for (int i = 0; i < width_2x; i++)
	{
		temp = p.alpha * (((i + 1.f) / width) - 1.f);
		temp *= temp;
		gauss_win[i] = expf(temp*(-0.5f));
	}

	gpuErrchk(cudaMalloc((void **)&d_gauss_win, (p.batchAscans * width_2x * sizeof(float))));
	gpuErrchk(cudaMemcpy(d_gauss_win, gauss_win, (width_2x * sizeof(float)), cudaMemcpyHostToDevice));
	repmat << <dimLine_w2, dimLine_B >> >(p.batchAscans, width_2x, d_gauss_win); gpuErrchk(cudaPeekAtLastError());

	delete[] gauss_win;

	cout << "	- Gauss win matrix created" << endl;
}

//load calibration file
void Data::loadFile(string fname, int length, float *result_array)
{
	//std::cout << __FUNCTION__ << " " << fname.c_str() << std::endl;
	ifstream is(fname.c_str());
	if (is)
	{
		for (int i = 0; i < length; i++)
		{
			is >> result_array[i];
			//std::cout<<result_array[i]<<"	";
		}
		//std::cout<<std::endl;
		is.close();
	}
	else
	{
		cerr << "File could not be opened!\n"; // Report error
		cerr << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("Calibration File Error");
	}
}


//add by Tao
///get ROI load initial and middle
void Data::loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int used_pixels, int height, float2 *result_array)
{
	//std::cout<<__FUNCTION__<<"	"<<fname<<std::endl;
	//cout << __FUNCTION__<< " startLocation " << startLocation << " all_pixels " << all_pixels << " used_pixels " << used_pixels<< " height " << height << endl;


	ifstream is(fname.c_str(), ios::in | ios::binary);
	if (is)
	{
		int x = 0;
		for (int h = 0; h < height; h++)
		{
			//std::cout<<"position:	"<<startLocation + all_pixels * h * 2<<std::endl;
			//is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			is.seekg(startLocation + (all_pixels * h) * 2, ios::beg);
			for (int i = 0; i < used_pixels; i++)
			{
				is.read(reinterpret_cast<char*>(&x), sizeof(uint16_t));
				//result_array[used_pixels * h + i].x = (float)x;
				result_array[used_pixels * h + i].x = (float)x;
				//std::cout<<x<<"	";
			}
			//std::cout<<std::endl;
		}
		is.close();
	}
	else
	{
		cerr << fname << std::endl;
		cerr << "File could not be opened!\n"; // Report error
		cerr << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("File Open Error");
	}
}



// add by Brian
//data.loadFile(fname, startLocation, all_pixels, used_pixels, height, result_array);
void Data::loadFile(string fname, int startLocation, int all_pixels, int used_pixels, int height, uint16_t *result_array)
{
	//std::cout<<__FUNCTION__<<"	"<<fname<<std::endl;
	//cout << __FUNCTION__<< " startLocation " << startLocation << " all_pixels " << all_pixels << " used_pixels " << used_pixels<< " height " << height << endl;

	ifstream is(fname.c_str(), ios::in | ios::binary);
	if (is)
	{
		for (int h = 0; h < height; h++)
		{
			//is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			is.seekg(startLocation + (all_pixels * h) * 2, ios::beg);
			//is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
			is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
		}
		is.close();
	}
	else
	{
		cerr << fname << std::endl;
		cerr << "File could not be opened!\n";
		cerr << "Error code: " << strerror(errno);
		throw invalid_argument("Data File Error");
	}
}

// add by Brian
//data.loadFile(fname, startLocation, all_pixels, used_pixels, height, result_array); for data process frame by frame
void Data::loadFile(string fname, long long startLocation, int all_pixels, int used_pixels, int height, uint16_t *result_array)
{
	//std::cout<<__FUNCTION__<<"	"<<fname<<std::endl;
	//cout << __FUNCTION__<< " startLocation " << startLocation << " all_pixels " << all_pixels << " used_pixels " << used_pixels<< " height " << height << endl;

	ifstream is(fname.c_str(), ios::in | ios::binary);
	if (is)
	{
		for (int h = 0; h < height; h++)
		{
			// is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			is.seekg(startLocation + (all_pixels * h) * 2, ios::beg);
			//is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
			is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
		}
		is.close();
	}
	else
	{
		cerr << fname << std::endl;
		cerr << "File could not be opened!\n";
		cerr << "Error code: " << strerror(errno);
		throw invalid_argument("Data File Error");
	}
}



// the following are the need for setting camera start pixel and end pixel

void Data::loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, uint16_t *result_array)
{
	int inner_used_pixel = camEndPixel - camStartPixel + 1;
	//std::cout << __FUNCTION__ << " startLocation " << startLocation << " all_pixels " << all_pixels << " height " << height << endl;
	ifstream is(fname.c_str(), ios::in | ios::binary);
	if (is)
	{
		for (int h = 0; h < height; h++)
		{
			// is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			is.seekg(startLocation + (all_pixels * h + camStartPixel - 1) * 2, ios::beg);
			//is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
			is.read(reinterpret_cast<char*>(result_array + inner_used_pixel * h), inner_used_pixel * 2);
		}
		is.close();
	}
	else
	{
		cerr << fname << std::endl;
		cerr << "File could not be opened!\n";
		cerr << "Error code: " << strerror(errno);
		throw invalid_argument("Data File Error");
	}
}

void Data::loadFile(std::string fname, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, float2 *result_array)
{
	int inner_used_pixel = camEndPixel - camStartPixel + 1;


	ifstream is(fname.c_str(), ios::in | ios::binary);
	if (is)
	{
		int x = 0;
		for (int h = 0; h < height; h++)
		{
			//std::cout<<"position:	"<<startLocation + all_pixels * h * 2<<std::endl;
			//is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			is.seekg(startLocation + (all_pixels * h + camStartPixel - 1) * 2, ios::beg);
			for (int i = 0; i < inner_used_pixel; i++)
			{
				is.read(reinterpret_cast<char*>(&x), sizeof(uint16_t));
				//result_array[used_pixels * h + i].x = (float)x;
				result_array[inner_used_pixel * h + i].x = (float)x;
				//std::cout<<x<<"	";
			}
			//std::cout<<std::endl;
		}
		is.close();
	}
	else
	{
		cerr << fname << std::endl;
		cerr << "File could not be opened!\n"; // Report error
		cerr << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("File Open Error");
	}
}

// the same with loadFil,used to load memory data

void Data::loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, uint16_t *result_array)
{
	int inner_used_pixel = camEndPixel - camStartPixel + 1;
	//std::cout << __FUNCTION__ << " startLocation " << startLocation << " all_pixels " << all_pixels << " height " << height << endl;
	uint16_t *cursor_ptr = mem_data;
	if (nullptr != mem_data)
	{
		for (int h = 0; h < height; h++)
		{
			// is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			//is.seekg(startLocation + (all_pixels * h + camStartPixel - 1) * 2, ios::beg);
			//is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
			//is.read(reinterpret_cast<char*>(result_array + inner_used_pixel * h), inner_used_pixel * 2);
			memcpy(result_array + inner_used_pixel * h, cursor_ptr + startLocation + (all_pixels * h + camStartPixel - 1), inner_used_pixel * sizeof(uint16_t));
		}
	}
	else
	{
		std::cerr << __FUNCTION__ << "Mem data could not be opened!\n"; // Report error
		std::cerr << __FUNCTION__ << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("Mem data did`t exist!");
	}
}
void Data::loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, int16_t *result_array)
{
	int inner_used_pixel = camEndPixel - camStartPixel + 1;
	//std::cout << __FUNCTION__ << " startLocation " << startLocation << " all_pixels " << all_pixels << " height " << height << endl;
	uint16_t *cursor_ptr = mem_data;
	if (nullptr != mem_data)
	{
		for (int h = 0; h < height; h++)
		{
			// is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			//is.seekg(startLocation + (all_pixels * h + camStartPixel - 1) * 2, ios::beg);
			//is.read(reinterpret_cast<char*>(result_array + used_pixels * h), used_pixels * 2);
			//is.read(reinterpret_cast<char*>(result_array + inner_used_pixel * h), inner_used_pixel * 2);
			memcpy(result_array + inner_used_pixel * h, cursor_ptr + startLocation + (all_pixels * h + camStartPixel - 1), inner_used_pixel * sizeof(uint16_t));
		}
	}
	else
	{
		std::cerr << __FUNCTION__ << "Mem data could not be opened!\n"; // Report error
		std::cerr << __FUNCTION__ << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("Mem data did`t exist!");
	}
}

void Data::loadMem(uint16_t *mem_data, unsigned long long startLocation, int all_pixels, int camStartPixel, int camEndPixel, int height, float2 *result_array)
{
	int inner_used_pixel = camEndPixel - camStartPixel + 1;
	uint16_t *cursor_ptr = nullptr;
	if (nullptr != mem_data)
	{
		int x = 0;
		for (int h = 0; h < height; h++)
		{
			//std::cout<<"position:	"<<startLocation + all_pixels * h * 2<<std::endl;
			//is.seekg(startLocation + all_pixels * h * 2,ios::beg);
			//is.seekg(startLocation + (all_pixels * h + camStartPixel - 1) * 2, ios::beg);
			cursor_ptr = mem_data + (startLocation + (all_pixels * h + camStartPixel - 1));
			for (int i = 0; i < inner_used_pixel; i++)
			{
				//is.read(reinterpret_cast<char*>(&x), sizeof(uint16_t));
				//result_array[used_pixels * h + i].x = (float)x;
				result_array[inner_used_pixel * h + i].x = (float)*cursor_ptr++;
				//std::cout<<x<<"	";
			}
			//std::cout<<std::endl;
		}
	}
	else
	{
		//cerr << fname << std::endl;
		std::cerr << __FUNCTION__ << "Mem data could not be opened!\n"; // Report error
		std::cerr << __FUNCTION__ << "Error code: " << strerror(errno); // Get some info as to why
		throw invalid_argument("Mem data did`t exist!");
	}
}

