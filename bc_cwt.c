#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#include "bc_denoise.h"

extern Param_t *Param;

// local functions
static void wfilth(fftw_complex *psih, double a, fftw_complex *fft_in);

static void wfilth(fftw_complex *psih, double a, fftw_complex *fft_in) {
	// Morlet wavelet
	double mu, cs, ks, w;

	mu = 2 * PI;
	cs = pow(1 + exp(-1. * mu * mu) - 2 * exp(-0.75 * mu * mu), -0.5) * pow(PI, -0.25) * sqrt(a) / sqrt(2. * PI);
	ks = exp(-0.5 * mu * mu);

	for (int i = 0; i < Param->dataLength; i++) {
		if (i <= Param->dataLength / 2) {
			w = i * a * 2. * PI / Param->dataLength;
		} else {
			w = (i - Param->dataLength) * a * 2. * PI / Param->dataLength;
		}

		psih[i] = cs * (exp(-0.5 * (mu - w) * (mu - w)) - ks * exp(-0.5 * w * w)) * fft_in[i] / Param->dataLength;

		if (i % 2 == 1) {
			psih[i] *= -1;
		}
	}
}

void cwt_forward() {
	fftw_complex *in, *fft_in, *psih, *ifft_psih;

	in          = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	fft_in      = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	psih        = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	ifft_psih   = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);

	if (in == NULL || fft_in == NULL || psih == NULL || ifft_psih == NULL) {
		fprintf(stderr, "Cannot allocated memory for fft!\n");
		exit(0);
	}

	for (int i = 0; i < Param->dataLength; i++) {
		in[i] = Param->data[i];
	}

	fftw_plan p = fftw_plan_dft_1d(Param->dataLength, in, fft_in, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan r = fftw_plan_dft_1d(Param->dataLength, psih, ifft_psih, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	Param->dataCWT = (fftw_complex *)malloc(sizeof(fftw_complex) * Param->numOct * Param->numVoice * Param->originalDataLength);
	if (Param->dataCWT == NULL) {
		fprintf(stderr, "Cannot allocated memory for cwt!\n");
		exit(0);
	}

	// for each octave
	for (int i = 0; i < Param->numOct * Param->numVoice; i++) {
		double a = pow(2, (i + 1.f) / Param->numVoice);

		// apply the wavelet
		wfilth(psih, a, fft_in);

		fftw_execute(r);

		for (int j = 0; j < Param->dataLength; j++) {
			if (j >= Param->leftTaper && j < Param->leftTaper + Param->originalDataLength) {
				int index = i * Param->originalDataLength + j - Param->leftTaper;
				if (j < Param->dataLength / 2) {
					Param->dataCWT[index] = ifft_psih[j + Param->dataLength / 2];
				} else {
					Param->dataCWT[index] = ifft_psih[j - Param->dataLength / 2];
				}
			}
		}
	}

	fftw_destroy_plan(p);
	fftw_destroy_plan(r);

	fftw_free(in);
	fftw_free(fft_in);
	fftw_free(psih);
	fftw_free(ifft_psih);
}

void cwt_inverse() {
	// this is only good for morlet wavelet
	double cpsi = 0.161252589430996 / 4. / PI;

	fftw_complex *in, *fft_in, *psih, *ifft_psih, *denoisedData;

	in           = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	fft_in       = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	psih         = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	ifft_psih    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->dataLength);
	denoisedData = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Param->originalDataLength);

	if (in == NULL || fft_in == NULL || psih == NULL || ifft_psih == NULL || denoisedData == NULL) {
		fprintf(stderr, "Cannot allocated memory for fft!\n");
		exit(0);
	}

	for (int i = 0; i < Param->dataLength; i++) {
		in[i] = 0;
		if (i < Param->originalDataLength) {
			denoisedData[i] = 0;
		}
	}

	fftw_plan p = fftw_plan_dft_1d(Param->dataLength, in, fft_in, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan r = fftw_plan_dft_1d(Param->dataLength, psih, ifft_psih, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (int i = 0; i < Param->numOct * Param->numVoice; i++) {
		double a = pow(2, (i + 1.f) / Param->numVoice);

		for (int j = 0; j <Param->originalDataLength; j++) {
			in[j + Param->leftTaper] = Param->dataCWT[i * Param->originalDataLength + j];
		}

		fftw_execute(p);

		wfilth(psih, a, fft_in);

		fftw_execute(r);

		for (int j = 0; j < Param->dataLength; j++) {
			if (j >= Param->leftTaper && j < Param->leftTaper + Param->originalDataLength) {
				if (j < Param->dataLength / 2) {
					denoisedData[j - Param->leftTaper] += ifft_psih[j + Param->dataLength / 2] / a;
				} else {
					denoisedData[j - Param->leftTaper] += ifft_psih[j - Param->dataLength / 2] / a;
				}
			}
		}
	}

	double c = log(2.) / cpsi / Param->numVoice;
	for (int j = 0; j < Param->originalDataLength; j++) {
		Param->data[j + Param->leftTaper] = c * creal(denoisedData[j]);
	}

	fftw_destroy_plan(p);
	fftw_destroy_plan(r);

	fftw_free(in);
	fftw_free(fft_in);
	fftw_free(psih);
	fftw_free(ifft_psih);
	fftw_free(denoisedData);
}