#include "helper.cuh"
#include <cmath>

using namespace std;

Helper::Helper(Parameters &p) : p(p)
{										// Batching affects dimensions. Width and frames are the same for each class. Height changes.
	height_1fr = p.numAscansPerBscan / p.batchFrames;	// height of 1 frame
	height_bfr = p.numAscansPerBscan;					// height of a batch of frames

	used_width = p.numUsedPixels;
	width = p.numCameraPixels;
	width_2x = p.numCameraPixels * 2;
	width_trm = p.endPixel - p.startPixel + 1;			// trim width. 
	//frames = p.numBScans;
	frames = p.numBScans * p.avgBscan;

										// Grid dimensions also change as height changes.
	dimLine_w = dim3((width+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, 1, 1);
	dimLine_w2 = dim3((width_2x+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, 1, 1);
	dimLine_wt = dim3((width_trm+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, 1, 1);

	dimLine_uw = dim3((used_width+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, 1, 1);
	dimLine_uw2 = dim3((used_width * 2+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, 1, 1);

	dimGrid_B = dim3(TILE_WIDTH, TILE_WIDTH, 1);		//kernel launch block size, 2d
	dimLine_B = dim3(THREADS_PER_BLOCK, 1, 1);			//kernel launch block size, 1d

	//w2Recip = 1.f/(float)width_2x;						//reciprocal of these is taken so that later we can multiply instead of divide
	w2Recip = 1.f/((float)used_width * 2);						//reciprocal of these is taken so that later we can multiply instead of divide
	grayRecip = 1.f/(float)p.grayLevel;					//
}

Helper::~Helper()
{
}

void Helper::gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  cout << "GPUassert: " << cudaGetErrorString(code) << " file:" << file << " line:" << line;
      if (abort) throw invalid_argument("CUDA Error");
   }
}

void Helper::cufftAssert(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) 
	{
		fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err,cudaGetErrorEnum(err));
		throw invalid_argument("CUFFT Error");
    }
}

const char* Helper::cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

void Helper::columnMean(int h, int w, float2 *my_array, float *result_array, float &columnMeanMax)
{	
	float sum = 0;
	float mean = 0;
	int count = 0;

	for (int j = 0; j < w; ++j)
	{
		for (int i = 0; i < h; ++i)
		{
			sum += my_array[i * w + j].x;
			count++;
		}
		mean = (float)sum / count;
		result_array[j] = mean;
		if (mean > columnMeanMax)
			columnMeanMax = mean;
		sum = 0;
		count = 0;
	}
}

void Helper::FFT(int h, int w, float2 *initial_array, float2 *result_array)
{
	int n[2] = {w, h};
	cufftHandle plan;
	cufftErrchk( cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,h) );
	cufftErrchk( cufftExecC2C(plan, initial_array, result_array, CUFFT_FORWARD) );
	cufftErrchk( cufftDestroy(plan) );
}

void Helper::IFT(int h, int w, dim3 dg, dim3 db, float2 *initial_array, float2 *result_array){
	
	int n[2] = {w, h};
	cufftHandle plan;
	cufftErrchk( cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,h) );
	cufftErrchk( cufftExecC2C(plan, initial_array, result_array, CUFFT_INVERSE) );
	cufftErrchk( cufftDestroy(plan) );

	float sclr = (1.f/w);
	scale_IFT<<<dg,db>>>(h, w, sclr, result_array); gpuErrchk( cudaPeekAtLastError() );
}

void Helper::output_csv(int height, int width, float *my_array, std::string flname)
{
	float *temp_array = new float[height*width];
	cudaMemcpy(temp_array, my_array, (height * width * sizeof(float)), cudaMemcpyDeviceToHost);

	cout << "OUTPUT begins" << endl;
	ofstream output_file(flname.c_str());

	for (int i = 0; i < height ; i++){
		for (int j = 0; j < width; j++)
			output_file << temp_array[i * width + j] << ",";
		output_file << endl;
	}
	output_file.close();
	cout << "OUTPUT ends" << endl;

	delete[] temp_array;
}

void Helper::output_csv(int height, int width, float2 *my_array, std::string flname)
{
	float2 *temp_array = new float2[height*width];
	cudaMemcpy(temp_array, my_array, (height * width * sizeof(float2)), cudaMemcpyDeviceToHost);

	cout << "OUTPUT begins" << endl;
	ofstream output_file(flname.c_str());

	for (int i = 0; i < height ; i++){
		for (int j = 0; j < width; j++)
			output_file << temp_array[i * width + j].x << ",";  // change to y for imaginary check.
		output_file << endl;
	}
	output_file.close();
	cout << "OUTPUT ends" << endl;

	delete[] temp_array;
}


void Helper::print_gpu_data(int height, int width, float2* gpu_data, string name)
{
	float2 *data = new float2[height * width]();
	gpuErrchk( cudaMemcpy(data, gpu_data, (height * width * sizeof(float2)), cudaMemcpyDeviceToHost) );
	ofstream outFile(name + ".csv", ofstream::out);
	for( int m = 0; m < height ; m++)
	{
		for( int n = 0; n < width; n++)
		{
			//if (data[m * width + n].y < 0) {
			//	outFile << data[m * width + n].x << ",";//<<data[m * width + n].y << "j ";
			//} else {
			//	outFile << data[m * width + n].x << ",";//"+"<<data[m * width + n].y << "j ";
			//}
			outFile << std::abs(data[m * width + n].x) << ",";
		}
		cout << endl;
	}
	delete[] data;
	outFile.close();
}

void Helper::print_gpu_data(int height, int width, uint16_t* gpu_data, string name)
{
	uint16_t *data = new uint16_t[height * width]();
	gpuErrchk( cudaMemcpy(data, gpu_data, (height * width * sizeof(uint16_t)), cudaMemcpyDeviceToHost) );

	cout << endl;
	cout << name<<endl;
	for( int m = 0; m < height ; m++)
	{
		for( int n = 0; n < width; n++)
		{
			cout<<data[m * width + n] << " ";
		}
		cout << endl;
	}
}

void Helper::print_gpu_data(int height, int width, float* gpu_data, string name)
{
	float *data = new float[height * width]();
	gpuErrchk(cudaMemcpy(data, gpu_data, (height * width * sizeof(float)), cudaMemcpyDeviceToHost));
	ofstream outFile(name + ".csv", ofstream::out);
	/*cout << endl;
	cout << name<<endl;*/
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			outFile << data[m * width + n] << ",";
		}
		outFile << endl;
	}
	outFile.close();
}

void Helper::print_cpu_data(int height, int width, float* cpu_data, string name)
{
	ofstream outFile(name + ".csv", ofstream::out);
	for( int m = 0; m < height ; m++)
	{
		for( int n = 0; n < width; n++)
		{
			outFile <<cpu_data[m * width + n] << ",";
		}
		outFile << endl;
	}
	outFile.close();
}
void Helper::print_cpu_data(int height, int width, int16_t* cpu_data, string name)
{
	ofstream outFile(name + ".csv", ofstream::out);
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			outFile << cpu_data[m * width + n] << ",";
		}
		outFile << endl;
	}
	outFile.close();
}

void Helper::print_cpu_data(int height, int width, uchar * cpu_data, string name)
{
	ofstream outFile(name + ".csv", ofstream::out);
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			outFile << (int)cpu_data[m * width + n] << ",";
		}
		outFile << endl;
	}
	outFile.close();
}

void Helper::print_cpu_data(int height, int width, float* cpu_data, string name, int id)
{
	cout << endl;
	cout << name<<endl;
	for( int m = 0; m < height ; m++)
	{
		for( int n = 0; n < width; n++)
		{
			cout<<cpu_data[m * width + n] << " ";
		}
		cout << endl;
	}
}

/*  //Some extra debug functions I wrote.

void output_csv(int height, int width, vector<float> &my_array){
	
	cout << "OUTPUT begins" << endl;
	ofstream output_file("C:\\Users\\ans915\\Desktop\\data\\testfile.csv");
	for (int i = 0; i < height ; i++){
		for (int j = 0; j < width; j++)
			output_file << my_array[i * width + j] << ",";
		output_file << endl;
	}
	output_file.close();
	cout << "OUTPUT ends" << endl;
}

void check_function(int height, int width, float *my_array){
	
	cout << "Check_function starts" << endl;

	vector<float> check_vector;
	vector<float> difference_vector(height * width);
    ifstream check_file;
    check_file.open ("C:\\Users\\ans915\\Documents\\MATLAB\\my_data.txt");
    float y;
    while (check_file >> y)
        check_vector.push_back(y);
    check_file.close();
	cout << "Check vector length is: " << check_vector.size() << endl;

	float max_value = 0;
	float min_value = 0;
	int count = 0;
	int min_index = 0;
	int max_index = 0;
	int max_error_occurences = 0;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){										     //change here (length of j, variable below)
			difference_vector[i * width + j] = check_vector[i * width + j] - my_array[i * width + j];
			
			if ((i == 0) && (j < 100)) {
				cout << j << " " << check_vector[i * width + j] << " " << my_array[i * width + j] << endl;
			}

			if (difference_vector[i * width + j] < min_value)
				min_value = difference_vector[i * width + j];
			if (difference_vector[i * width + j] > max_value)
				max_value = difference_vector[i * width + j];
			if ((difference_vector[i * width + j] == min_value) || (difference_vector[i * width + j] == max_value))
				max_error_occurences++;				
			if (difference_vector[i * width + j] > difference_vector[max_index])
				max_index = i * width + j;              
			if (difference_vector[i * width + j] < difference_vector[min_index])
				min_index = i * width + j; 
			count++;
		}
	}

	cout << endl << "Max difference value is: " << max_value << endl;
	cout << "Max difference occurs at index: " << max_index << endl;
	cout << "Min difference value is: " << min_value << endl;
	cout << "Min difference occurs at index: " << min_index << endl;
	cout << "Occurences: " << max_error_occurences << endl;
	cout << "Values compared: " << count << endl << endl;

	system("pause");
}

void check_function(int height, int width, float2 *my_array){
	
	cout << "Check_function starts" << endl;

	vector<float> check_vector;
	vector<float> difference_vector(height * width);
	
    ifstream check_file;
    check_file.open ("C:\\Users\\ans915\\Documents\\MATLAB\\my_data.txt");
    float y;
    while (check_file >> y)
        check_vector.push_back(y);
    check_file.close();
	cout << "Check vector length is: " << check_vector.size() << endl;
	//cout << "cpp vector length is: " << my_array.size()  << endl;					//change here

	float max_value = 0;
	float min_value = 0;
	int count = 0;
	int min_index = 0;
	int max_index = 0;
	int max_error_occurences = 0;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){										     //change here (length of j, variable below)
			difference_vector[i * width + j] = check_vector[i * width + j] - my_array[i * width + j].x;
			
			if ((i == 0) && (j < 100)) {
				cout << j << " " << check_vector[i * width + j] << " " << my_array[i * width + j].x << endl;
			}

			if (difference_vector[i * width + j] < min_value){
				min_value = difference_vector[i * width + j];

			}
			if (difference_vector[i * width + j] > max_value)
				max_value = difference_vector[i * width + j];
			if ((difference_vector[i * width + j] == min_value) || (difference_vector[i * width + j] == max_value))
				max_error_occurences++;				
			if (difference_vector[i * width + j] > difference_vector[max_index])
				max_index = i * width + j;              
			if (difference_vector[i * width + j] < difference_vector[min_index])
				min_index = i * width + j; 
			count++;
		}
	}
	float min_percent_error = 0;
	float max_percent_error = 0;
	float current_percent_error = 0;

	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			current_percent_error = difference_vector[i * width + j] / my_array[i * width + j].x * 100;
			if (current_percent_error < min_percent_error)
				min_percent_error = current_percent_error;
			if (current_percent_error > max_percent_error)
				max_percent_error = current_percent_error;
		}
	}

	cout << endl << "Max difference value is: " << max_value << " at index: " << max_index << endl;
	cout << "Min difference value is: " << min_value << " at index: " << min_index << endl;
	cout << "Occurences: " << max_error_occurences << endl;
	cout << "Values compared: " << count << endl << endl << endl;

	cout << "Min percent error is: " << min_percent_error << endl;
	cout << "Max percent error is: " << max_percent_error << endl;


	system("pause");
}

void check_function(int height, int width, vector<float> &my_array){
	
	cout << "Check_function starts" << endl;

	vector<float> check_vector;
	vector<float> difference_vector(height * width);
    ifstream check_file;
    check_file.open ("C:\\Users\\ans915\\Documents\\MATLAB\\my_data.txt");
    float y;
    while (check_file >> y)
        check_vector.push_back(y);
    check_file.close();
	cout << "Check vector length is: " << check_vector.size() << endl;
	cout << "cpp vector length is: " << my_array.size()  << endl;					//change here

	float max_value = 0;
	float min_value = 0;
	int count = 0;
	int min_index = 0;
	int max_index = 0;
	int max_error_occurences = 0;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){										     //change here (length of j, variable below)
			difference_vector[i * width + j] = check_vector[i * width + j] - my_array[i * width + j];
			
			if ((i == 0) && (j < 100)) {
				cout << j << " " << check_vector[i * width + j] << " " << my_array[i * width + j] << endl;
			}

			if (difference_vector[i * width + j] < min_value)
				min_value = difference_vector[i * width + j];
			if (difference_vector[i * width + j] > max_value)
				max_value = difference_vector[i * width + j];
			if ((difference_vector[i * width + j] == min_value) || (difference_vector[i * width + j] == max_value))
				max_error_occurences++;				
			if (difference_vector[i * width + j] > difference_vector[max_index])
				max_index = i * width + j;              
			if (difference_vector[i * width + j] < difference_vector[min_index])
				min_index = i * width + j; 
			count++;
		}
	}

	cout << endl << "Max difference value is: " << max_value << endl;
	cout << "Max difference occurs at index: " << max_index << endl;
	cout << "Min difference value is: " << min_value << endl;
	cout << "Min difference occurs at index: " << min_index << endl;
	cout << "Occurences: " << max_error_occurences << endl;
	cout << "Values compared: " << count << endl << endl;

	system("pause");
}

*/
