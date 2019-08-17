# -*- Makefile -*-
#
# Makefile for bc_denoise
#
# Author: Yang Yang <Yang.Yang@memphis.edu>
#

## usegpu; 1: compile the GPU version; otherwise, use CPU version
usegpu = 1

CC           = gcc
CFLAGS       = -Wall -O3
LD           = $(CC)
LDFLAGS      = -lm -lstdc++

##fftw3 library
FFTW3_INCLUDE = /public/apps/fftw/3.3.8/include
FFTW3_LIB = /public/apps/fftw/3.3.8/lib

CFLAGS += -I${FFTW3_INCLUDE}
LDFLAGS += -L${FFTW3_LIB} -lfftw3

ifeq ($(usegpu), 1)
	# cuda compiler and library
	NVCC = nvcc
	CUDA_LIB = /cm/shared/apps/cuda92/toolkit/9.2.88/lib64

	CFLAGS += -DUSE_GPU
	LDFLAGS += -L${CUDA_LIB} -lcudart -lcufft

	TARGET = bc_denoise_gpu
else
	TARGET = bc_denoise_cpu
endif

#
# Object modules 
#
OBJECTS = bc_denoise.o bc_cwt.o sac.o bc_process.o bc_parameter.o bc_util.o

ifeq ($(usegpu), 1)
	OBJECTS += cudaKernel.o
endif

.PHONY: all tags clean cleanall

all: $(TARGET)

${TARGET}: ${OBJECTS}
		$(LD) $^ \
		$(LDFLAGS) \
		-o $@

ifeq ($(usegpu), 1)
cudaKernel.o: cudaKernel.cu
	$(NVCC) $(LDFLAGS) -c cudaKernel.cu
endif

clean:
		rm -f $(OBJECTS) *.o

cleanall:
		rm -f $(OBJECTS) $(TARGET) *.o
