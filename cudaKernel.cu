#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define THREADS_PER_BLOCK 1024
#define PI 3.1415926535857932384626433f

#define FULL_MASK 0xffffffff

typedef float2 Complex;

Complex *h_data;
Complex *d_data;
Complex *d_dataCWT;
float   *d_noise_max;
float   *d_noise_min;

__device__ inline void floatAtomicAdd (float *address, float value) {
	int oldval, newval, readback;

	oldval = __float_as_int(*address);
	newval = __float_as_int(__int_as_float(oldval) + value);
	while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
		oldval = readback;
		newval = __float_as_int(__int_as_float(oldval) + value);
	}
}

__device__ inline void floatAtomicMin(float *address, float value) {
	int oldval, newval, readback;

	oldval = __float_as_int(*address);
	newval = __float_as_int(fminf(__int_as_float(oldval), value));
	while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
		oldval = readback;
		newval = __float_as_int(fminf(__int_as_float(oldval), value));
	}
}

__device__ inline void floatAtomicMax(float *address, float value) {
	int oldval, newval, readback;

	oldval = __float_as_int(*address);
	newval = __float_as_int(fmaxf(__int_as_float(oldval), value));
	while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
		oldval = readback;
		newval = __float_as_int(fmaxf(__int_as_float(oldval), value));
	}
}

static __global__ void complex_wfilith(Complex* dataCWT, Complex *data, int length, int voice) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    float mu, cs, ks, w, a, waveletFFT;

    if (j < length) {
    	int index = i * length + j;

    	a = powf(2, (i + 1.f) / voice);
		mu = 2 * PI;
		cs = powf(1 + exp(-1.f * mu * mu) - 2 * expf(-0.75f * mu * mu), -0.5f) * powf(PI, -0.25f) * sqrtf(a) / sqrtf(2.f * PI);
		ks = expf(-0.5f * mu * mu);
		if (j <= length / 2) {
			w = j * a * 2. * PI / length;
		} else {
			w = (j - length) * a * 2.f * PI / length;
		}

		waveletFFT = cs * (expf(-0.5f * (mu - w) * (mu - w)) - ks * expf(-0.5f * w * w)) / length;

    	dataCWT[index].x = data[j].x * waveletFFT;
    	dataCWT[index].y = data[j].y * waveletFFT;
    }
}

static __global__ void noise_model(Complex* dataCWT, float* max, float *min, int ib, int ie, int length) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
	
	if (j == 0) {
		max[i] = 0;
		min[i] = 1e40;
	}

	__syncthreads();

	float maxd = 0;
	float mind = 1e40;
	if (j >= ib && j <= ie) {
    	int index = i * length + j;

    	maxd = sqrtf(dataCWT[index].x * dataCWT[index].x + dataCWT[index].y * dataCWT[index].y);
    	mind = maxd;
    }

    for (int i = 16; i; i >>= 1) {
		maxd = fmaxf(__shfl_down_sync(FULL_MASK, maxd, i), maxd);
		mind = fminf(__shfl_down_sync(FULL_MASK, mind, i), mind);
	}

	__shared__ float blockmaxd;
	__shared__ float blockmind;
	blockmaxd = 0;
	blockmind = 1e40;

	__syncthreads();

	if (threadIdx.x % 32 == 0) {
		floatAtomicMax(&blockmaxd, maxd);
		floatAtomicMin(&blockmind, mind);
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		floatAtomicMax(max + i, blockmaxd);
		floatAtomicMin(min + i, blockmind);
	}
}

static __global__ void designal(Complex* dataCWT, int dataLength, float *max, float* min) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    if (j < dataLength) {
    	int index = i * dataLength + j;

    	float noiseLevel = min[i] + (max[i] - min[i]) * 0.99;
    	float amp = sqrtf(dataCWT[index].x * dataCWT[index].x + dataCWT[index].y * dataCWT[index].y);

    	if (amp > noiseLevel) {
    		dataCWT[index].x *= noiseLevel / amp;
    		dataCWT[index].y *= noiseLevel / amp;
    	}
    }
}

static __global__ void denoise(Complex* dataCWT, int dataLength, float *max, float* min) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    if (j < dataLength) {
    	int index = i * dataLength + j;

    	float noiseLevel = min[i] + (max[i] - min[i]) * 0.99;
    	float amp = sqrtf(dataCWT[index].x * dataCWT[index].x + dataCWT[index].y * dataCWT[index].y);

    	if (amp > noiseLevel) {
    		dataCWT[index].x *= (amp - noiseLevel) / amp;
    		dataCWT[index].y *= (amp - noiseLevel) / amp;
    	} else {
    		dataCWT[index].x = 0.f;
    		dataCWT[index].y = 0.f;
    	}
    }
}

static __global__ void complex_wfilith_inverse(Complex* dataCWT, int length, int voice) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    float mu, cs, ks, w, a, waveletFFT;

    if (j < length) {
    	int index = i * length + j;

    	a = powf(2, (i + 1.f) / voice);
		mu = 2.f * PI;
		cs = powf(1.f + exp(-1.f * mu * mu) - 2 * expf(-0.75f * mu * mu), -0.5f) * powf(PI, -0.25f) * sqrtf(a) / sqrtf(2.f * PI);
		ks = expf(-0.5f * mu * mu);
		if (j <= length / 2) {
			w = j * a * 2. * PI / length;
		} else {
			w = (j - length) * a * 2.f * PI / length;
		}

		waveletFFT = cs * (expf(-0.5f * (mu - w) * (mu - w)) - ks * expf(-0.5f * w * w)) / length / a * logf(2.f) / (0.161252589430996f / 4.f / PI) / voice;

    	dataCWT[index].x *= waveletFFT;
    	dataCWT[index].y *= waveletFFT;
    }
}

static __global__ void inverse_cwt_sum(Complex* dataCWT, Complex* data, int length) {
	//gloabl memory index
    int i = blockIdx.x;
    int j = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    if (j < length) {
    	int index = i * length + j;
    	floatAtomicAdd(&data[j].x, dataCWT[index].x);
    }
}

extern "C"
void cuda_cwt_forward(double *data, int dataLength, int numOct, int numVoice) {
	cufftHandle forwardPlan;
	cufftHandle inversePlan;

	// initial the GPU memory
	if (cudaMalloc((void**)&d_data,        sizeof(Complex) * dataLength)                     != cudaSuccess ||
		cudaMalloc((void**)&d_noise_max,   sizeof(float)   * numOct * numVoice)              != cudaSuccess ||
		cudaMalloc((void**)&d_noise_min,   sizeof(float)   * numOct * numVoice)              != cudaSuccess ||
		cudaMalloc((void**)&d_dataCWT,     sizeof(Complex) * numOct * numVoice * dataLength) != cudaSuccess) {
		fprintf(stderr, "Error allocate memory for GPU parameters\n");
		exit(0);
	}

	h_data = (Complex *)malloc(sizeof(Complex) * dataLength);
	if (h_data == NULL) {
		fprintf(stderr, "Cannot allocate h_data\n");
		exit(0);
	}

	for (int i = 0; i < dataLength; i++) {
		h_data[i].x = data[i];
		h_data[i].y = 0;
	}

	if (cudaMemcpy(d_data, h_data, sizeof(Complex) * dataLength, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Cannot copy from CPU to GPU\n");
		exit(0);
	}

	cufftPlan1d(&forwardPlan, dataLength, CUFFT_C2C, 1);

	cufftPlan1d(&inversePlan, dataLength, CUFFT_C2C, numOct * numVoice);

	cufftExecC2C(forwardPlan, (cufftComplex *)d_data, (cufftComplex *)d_data, CUFFT_FORWARD);

	complex_wfilith<<<dim3(numOct * numVoice, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, d_data, dataLength, numVoice);

	cufftExecC2C(inversePlan, (cufftComplex *)d_dataCWT, (cufftComplex *)d_dataCWT, CUFFT_INVERSE);

	cufftDestroy(forwardPlan);
	cufftDestroy(inversePlan);

}

extern "C"
void cuda_initial_noise_model(int ib, int ie, int dataLength, int length) {
	noise_model<<<dim3(length, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, d_noise_max, d_noise_min, ib, ie, dataLength);
}

extern "C"
void cuda_soft_thresholding(int dataLength, int length, int mode) {
	if (mode == 0) {
		designal<<<dim3(length, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, dataLength, d_noise_max, d_noise_min);
	} else {
		denoise<<<dim3(length, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, dataLength, d_noise_max, d_noise_min);
	}
}

extern "C"
void cuda_cwt_inverse(double *data, int dataLength, int numOct, int numVoice) {
	cufftHandle plan;

	if (cudaMemset(d_data, 0, sizeof(Complex) * dataLength) != cudaSuccess) {
		fprintf(stderr, "Cannot set memory to zeros\n");
		exit(0);
	}

	cufftPlan1d(&plan, dataLength, CUFFT_C2C, numOct * numVoice);

	cufftExecC2C(plan, (cufftComplex *)d_dataCWT, (cufftComplex *)d_dataCWT, CUFFT_FORWARD);

	complex_wfilith_inverse<<<dim3(numOct * numVoice, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, dataLength, numVoice);

	cufftExecC2C(plan, (cufftComplex *)d_dataCWT, (cufftComplex *)d_dataCWT, CUFFT_INVERSE);

	inverse_cwt_sum<<<dim3(numOct * numVoice, dataLength/THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(d_dataCWT, d_data, dataLength);

	if (cudaMemcpy(h_data, d_data, sizeof(Complex) * dataLength, cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "Cannot copy from GPU to CPU\n");
		exit(0);
	}

	for (int i = 0; i < dataLength; i++) {
		data[i] = h_data[i].x;
	}

	cufftDestroy(plan);

	free(h_data);

	cudaFree(d_data);
	cudaFree(d_dataCWT);
	cudaFree(d_noise_max);
	cudaFree(d_noise_min);
}