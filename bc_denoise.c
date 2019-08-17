#include <stdio.h>
#include <stdlib.h>

#include "bc_denoise.h"
#include "sac.h"

extern Param_t *Param;

int main(int argc, char ** argv) {
	SACHEAD sachead;
	float*  sacdata;
	char dataFileName[256];
	char outputDataFileName[256];

	char command[534];

	Param = (Param_t *)malloc(sizeof(Param_t));
	if (Param == NULL) {
		bc_abort(__func__, "malloc() fail", "Error allocate memory for parameters\n");
	}

	//init parameters from the input file
    parameter_init(argc, argv);

	// get all the files that need to be processed
	sprintf(command, "ls %s/%s > filename.list_bc", Param->inputDir, Param->inputFile);
	system(command);

	FILE *fileList;

	fileList = fopen("filename.list_bc", "r");
	if (fileList == NULL) {
		bc_abort(__func__, "fopen() fail", "Error open the filename list\n");
	}	

	//read file line by line
	while (fgets(dataFileName, sizeof(dataFileName), fileList) != NULL) {
		fprintf(stdout, "Processing file %s", dataFileName);

		// remove the enter at the end
		for (int ic = 0; ; ic++) {
			if (dataFileName[ic] == '\n') {
				dataFileName[ic] = '\0';
				break;
			}
		}

		sacdata = read_sac(dataFileName, &sachead);
		if (sacdata == NULL) {
			bc_abort(__func__, "read_sac fail", "Cannot read sac file %s\n", dataFileName);
		}

		// sac data information
		Param->dt = sachead.delta;
		Param->originalDataLength = sachead.npts;

		// estimate noise segment
		if (Param->mode == 0) {
			float minMax = 1e10;
			int numInOneSeg = sachead.npts / Param->nSeg;
			// ignore the first and last segments
			for (int iseg = 1; iseg < Param->nSeg - 1; iseg++) {
				float tmpMax = 0;
				for (int ip = 0; ip < numInOneSeg; ip++) {
					if (sacdata[iseg * numInOneSeg + ip] > tmpMax) {
						tmpMax = sacdata[iseg * numInOneSeg + ip];
					}
				}

				if (tmpMax < minMax) {
					minMax = tmpMax;
					Param->noiseStart = iseg * numInOneSeg;
					Param->noiseEnd   = (iseg + 1) * numInOneSeg - 1;
				}
			}
		} else {
			Param->noiseStart = (int)((Param->noiseStartTime - sachead.b) / sachead.delta);
			Param->noiseEnd = (int)((Param->noiseEndTime - sachead.b) / sachead.delta);

			if (Param->noiseStart < 0 || Param->noiseEnd > sachead.npts - 1) {
				bc_abort(__func__, "parameters error", "Invalid noise segments\n");
			}
		}

		int n = sachead.npts;
		Param->numOct = 0;
		while (n > 0) {
			n >>= 1;
			Param->numOct++;
		}

		Param->dataLength = 1 << Param->numOct;
		if (Param->dataLength / sachead.npts == 2) {
			Param->dataLength >>= 1;
		}

		Param->numOct--;

		Param->data = (double *)malloc(sizeof(double) * Param->dataLength);
		if (Param->data == NULL) {
			bc_abort(__func__, "parameters error", "Cannot allocate data\n");
		}

		Param->leftTaper = (Param->dataLength - sachead.npts) / 2;
		for (int i = 0; i < Param->dataLength; i++) {
			if (i < Param->leftTaper) {
				Param->data[i] = 0.;
			} else if (i < sachead.npts + Param->leftTaper) {
				Param->data[i] = sacdata[i - Param->leftTaper];
			} else {
				Param->data[i] = 0.;
			}
		}

#ifdef USE_GPU
		cuda_cwt_forward(Param->data, Param->dataLength, Param->numOct, Param->numVoice);

		cuda_initial_noise_model(Param->noiseStart + Param->leftTaper, Param->noiseEnd + Param->leftTaper, Param->dataLength, Param->numOct * Param->numVoice);

		cuda_soft_thresholding(Param->dataLength, Param->numOct * Param->numVoice, Param->mode);

		cuda_cwt_inverse(Param->data, Param->dataLength, Param->numOct, Param->numVoice);
#else
		cwt_forward();

		initial_noise_model();

		soft_thresholding();

		cwt_inverse();
#endif

		for (int i = 0; i < Param->originalDataLength; i++) {
			sacdata[i] = Param->data[i + Param->leftTaper];
		}

		// save denoised sac file
		sprintf(outputDataFileName, "%s%s", dataFileName, Param->outputAppend);
		write_sac(outputDataFileName, sachead, sacdata);

		free(sacdata);
		delete_parameters();
	}

	free(Param);

	fclose(fileList);

	// remove tmp files
	system("rm filename.list_bc");

	return 0;
}
