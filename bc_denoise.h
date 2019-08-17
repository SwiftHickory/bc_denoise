#ifndef __BC_DENOISE__H__
#define __BC_DENOISE__H__

#include <complex.h>
#include <fftw3.h>

#define PI 3.1415926535857932384626433

typedef struct Param_t {
	// file informations
	char theParametersInputFileName[256];
	char inputDir[256];
	char inputFile[256];
	char outputAppend[256];

	// desingal information
	int mode;
	int numVoice;
	int nSeg;
	double noiseStartTime;
	double noiseEndTime;
	int noiseStart;
	int noiseEnd;

	// data related
	int originalDataLength;
	int dataLength;
	int leftTaper;
	double dt;
	double *data;
	fftw_complex *dataCWT;
	double *noiseLevel;
	int numOct;

} Param_t;

Param_t *Param;

// bc_cwt.c
void cwt_forward();
void cwt_inverse();

// bc_process.c
void initial_noise_model();
void soft_thresholding();

// bc_util.c
int bc_abort(const char* fname, const char* perror_msg, const char* format, ...);
int parsetext(FILE* fp, const char* querystring, const char type, void* result);

// bc_parameter.c
int parameter_init(int argc, char** argv);
void delete_parameters();

#ifdef USE_GPU
// cudaKernel.cu
void cuda_cwt_forward(double *data, int dataLength, int numOct, int numVoice);
void cuda_cwt_inverse(double *data, int dataLength, int numOct, int numVoice);
void cuda_initial_noise_model(int ib, int ie, int dataLength, int length);
void cuda_soft_thresholding(int dataLength, int length, int mode);
#endif

#endif