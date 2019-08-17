README.txt

***********
Version 1.0
July 3, 2019

C (CPU/GPU) program for soft threshold CWT ambient noise designal and empirical Green's function denoise

Yang Yang
Yang.Yang@memphis.edu

The fftw3 library is needed, which can be downloaded from: http://www.fftw.org.
For the GPU version, NVidia CUDA is needed.

Change usegpu in the Makefile to compile CPU or GPU version

Prepare the input file parameters.in before running. Wildcard supported for multiple files (e.g. *.sac)
To run the code, type ./bc_denoise_cpu(or ./bc_denoise_gpu) parameters.in

******************************************************************************************
Disclaimer

This software is provided "as is" and can be freely used.  There are no claims that it is error free or will be useful for your particular application. Use at your own risk.
******************************************************************************************