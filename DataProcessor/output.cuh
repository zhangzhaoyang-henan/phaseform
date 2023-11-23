#pragma once
//#include "tiffio.h"

#include "background.cuh"
#include "TiffHandler.h"
#include <vector>

/*
 * Output class runs the actual data processing and handles writing to disk
 */

class Output : public Helper
{
public:
	Output(Parameters &p, Data &da, Interpolation &i, Dispersion &di, Background &b);
	~Output(void);

	//void process();
	void process(std::string datatype, std::vector<uint8_t>& bmpRawData, std::vector<float>& previewFFTrawData);
	static void shiftAndAvg(std::vector<cv::Mat>& mat_vec);
	void edgeCorrect(float *aloneAscan, int diff, int width);
	void flatOut(cv::Mat &srcImage);
	void CorrectDiff(cv::Mat &srcImage);
	void blurDiff(std::vector<int> &diff);
private:
	std::vector<int> diffs;
	Data &data;
	Interpolation &interp;
	Dispersion &disp;
	Background &bg;

	int16_t *h_buff_1, *h_buff_2, *h_buff_3, *h_buff_4;											// Various buffer arrays
	int16_t *d_raw_data_buf_1, *d_raw_data_buf_2, *d_raw_data_buf_3, *d_raw_data_buf_4;											// Various buffer arrays
	float2 *d_proc_buff_0, *d_proc_buff_1, *d_proc_buff_2, *d_proc_buff_trm;	// used to process data.
	float2 *d_proc_buff_3, *d_proc_buff_4, *d_proc_buff_5, *d_proc_buff_trm_1;	// used to process data.
	float *d_proc_buff_db, *d_proc_buff_trns;									//
	float *d_proc_buff_db_1, *d_proc_buff_trns_1;									//
	float *largeBuff;
	int height_ba, width_ba, width_2xba, frames_tot;							// Check documentation for explanation
	// of different dimension and kernel
	int used_width_ba, used_width_2xba;
	dim3 dimLine_wba, dimLine_w2xba, dimLine_wtba;								// launch parameters
	dim3 dimGrid_uw, dimLine_uwba;
	cufftHandle	plan_w, plan_w2, plan_w2_s2, plan_w_s2;
	cudaStream_t stream1, stream2;
	unsigned long long activeOffset;

	//handler for TIff
	UltraOCT::TiffHandler tiffhandler;
	UltraOCT::Size pageSize;

	void initResources();
	void processData(int it, float *proc_data_piece, std::vector<float>& previewFFTrawData);
	//void writeToDisk(float *proc_data_array);
	void writeToDisk(float *proc_data_array, std::string datatype, std::vector<uint8_t>& bmpRawData);	// reslicing happens here.
	void freeResources();

	size_t mem_avail, mem_total; // used for memory debug

	//flag to tell preview and acquire
	std::string process_type; // "preview" or "acquire"
};
