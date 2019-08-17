#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "bc_denoise.h"

extern Param_t *Param;

void initial_noise_model() {
	int bIndex = Param->noiseStart;
	int eIndex = Param->noiseEnd;

	double max, min, tmp;

	Param->noiseLevel = (double *)malloc(sizeof(double) * Param->numOct * Param->numVoice);
	if (Param->noiseLevel == NULL) {
		bc_abort(__func__, "malloc error", "Cannot allocate noise level\n");
	}

	// estimate noise level
	for (int i = 0; i < Param->numOct * Param->numVoice; i++) {
		max = 0;
		min = 1e10;
		for (int j = bIndex; j <= eIndex; j++) {
			tmp = cabs(Param->dataCWT[i * Param->originalDataLength + j]);
			if (tmp < min)
				min = tmp;
			if (tmp > max)
				max = tmp;
		}
		
		Param->noiseLevel[i] = (max - min) * 0.99 + min;
	}
}

void soft_thresholding() {
	if (Param->mode == 0) {
		// designal
		for (int i = 0; i < Param->numOct * Param->numVoice; i++) {
			for (int j = 0; j < Param->originalDataLength; j++) {
				int index = i * Param->originalDataLength + j;

				if (cabs(Param->dataCWT[index]) > Param->noiseLevel[i]) {
					Param->dataCWT[index] *= Param->noiseLevel[i] / cabs(Param->dataCWT[index]);
				}
			}
		}
	} else {
		// denoise
		for (int i = 0; i < Param->numOct * Param->numVoice; i++) {
			for (int j = 0; j < Param->originalDataLength; j++) {
				int index = i * Param->originalDataLength + j;
				
				if (cabs(Param->dataCWT[index]) > Param->noiseLevel[i]) {
					Param->dataCWT[index] *= (cabs(Param->dataCWT[index]) - Param->noiseLevel[i]) / cabs(Param->dataCWT[index]);
				} else {
					Param->dataCWT[index] = 0;
				}
			}
		}
	}
}