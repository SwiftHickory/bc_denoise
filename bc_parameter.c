#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bc_denoise.h"

extern Param_t* Param;

//local functions
static int parameter_read();

int parameter_init(int argc, char** argv) {
    //xdfwi running syntax
    if (argc != 2) {
        fprintf(stderr, "Usage: ./bc_denoise <parameters.in>\n\n");
        exit(1);
    }

    //get the input parameter file name
    strcpy(Param->theParametersInputFileName, argv[1]);

    fprintf(stdout, "Start to init parameters.\n");

    if (parameter_read() != 0) {
        bc_abort(__func__, "parameter_read() fail", "Error init input parameters\n");
    }

    return 0;
}

/**
 * initialize parameter from input file
 */
static int parameter_read() {
    FILE   *fp;
    char   tmp_char[256]; 
    int    tmp_int;
    double tmp_double;

    //obtain the specficiation of the simulation
    fp = fopen(Param->theParametersInputFileName, "r");
    if (fp == NULL) {
        bc_abort(__func__, "fopen() fail", "Error open the input parameter file: %s\n", Param->theParametersInputFileName);
    }

    if (parsetext(fp, "input_file_directory", 's', tmp_char) != 0)
        strcpy(Param->inputDir, "./");
    else
        strcpy(Param->inputDir, tmp_char);

    if (parsetext(fp, "input_filename_wildcard", 's', tmp_char) != 0)
        strcpy(Param->inputFile, "*.sac");
    else
        strcpy(Param->inputFile, tmp_char);

    if (parsetext(fp, "output_filename_append", 's', tmp_char) != 0)
        strcpy(Param->outputAppend, ".designaled");
    else
        strcpy(Param->outputAppend, tmp_char);

    if (parsetext(fp, "designal_or_denoise", 'i', &tmp_int) != 0)
        Param->mode = 0;
    else
        Param->mode = tmp_int;

    if (parsetext(fp, "number_of_voice", 'i', &tmp_int) != 0)
        Param->numVoice = 16;
    else
        Param->numVoice = tmp_int;

    if (Param->mode == 1) {
        if (parsetext(fp, "noise_start_time", 'd', &tmp_double) != 0) {
            fprintf(stderr, "Need noiseStartTime\n");
            return -1;
        } else {
            Param->noiseStartTime = tmp_double;
        }

        if (parsetext(fp, "noise_end_time", 'd', &tmp_double) != 0) {
            fprintf(stderr, "Need noiseEndTime\n");
            return -1;
        } else {
            Param->noiseEndTime = tmp_double;
        }
            
    } else {
        if (parsetext(fp, "number_of_segment", 'i', &tmp_int) != 0)
            Param->nSeg = 48;
        else
            Param->nSeg = tmp_int;
    }

    /* Sanity check */
    if (Param->numVoice < 4) {
        fprintf(stderr, "Number of voice is too small %d\n", Param->numVoice);
        return -1;
    }

    if (Param->mode == 1 && Param->noiseEndTime <= Param->noiseStartTime) {
        fprintf(stderr, "Noise start time should less than noise end time\n");
        return -1;
    }

    if (Param->mode == 0 && Param->nSeg < 0) {
        fprintf(stderr, "Number of segment should be positive %d\n", Param->nSeg);
        return -1;
    }

    return 0;
}

void delete_parameters() {
#ifndef USE_GPU
    fftw_free(Param->dataCWT);
    free(Param->noiseLevel);
#endif
    
    free(Param->data);
}
