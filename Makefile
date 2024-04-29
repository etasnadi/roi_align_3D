TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CCSRC=ccsrc
BUILD=build

all: ${BUILD}/libroi_align_cpu.so ${BUILD}/libroi_align_cpu_cuda.so

# 1. Compilation when CUDA toolkit is installed

# A: compile cuda sources into separate objects (-dc needed), 
# B: link device code from the result of A to a single object
# C: ordinary link the result of A and result of B into a single library
# D: compile the tensorflow host code and link with the result of C to get a shared library

# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
# https://forums.developer.nvidia.com/t/separate-compilation-of-cuda-code-into-library-for-use-with-existing-code-base/50774/7

# Code that is used on both device and host (called from host code and device code)
${BUILD}/roi_align.o: ${CCSRC}/roi_align.cc
	nvcc -c -o ${BUILD}/roi_align.o ${CCSRC}/roi_align.cc \
	${TF_CFLAGS} -dc -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

# A
${BUILD}/kernel_roi_align_grad.cu.o: ${CCSRC}/kernel_roi_align_grad.cu.cc ${CCSRC}/kernel_roi_align_grad.h
	nvcc -c -o ${BUILD}/kernel_roi_align_grad.cu.o \
	${CCSRC}/kernel_roi_align_grad.cu.cc \
	${TF_CFLAGS} -dc -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

# A
${BUILD}/kernel_roi_align.cu.o: ${CCSRC}/kernel_roi_align.cu.cc ${CCSRC}/kernel_roi_align.h
	nvcc -c -o ${BUILD}/kernel_roi_align.cu.o \
	${CCSRC}/kernel_roi_align.cu.cc \
	${TF_CFLAGS} -dc -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

# B
${BUILD}/devicelink.o: ${BUILD}/kernel_roi_align.cu.o ${BUILD}/kernel_roi_align_grad.cu.o ${BUILD}/roi_align.o
	nvcc -dlink -o ${BUILD}/devicelink.o ${BUILD}/kernel_roi_align.cu.o ${BUILD}/kernel_roi_align_grad.cu.o ${BUILD}/roi_align.o \
	-Xcompiler -fPIC

# C
${BUILD}/ordlink.a: ${BUILD}/kernel_roi_align_grad.cu.o ${BUILD}/roi_align.o ${BUILD}/devicelink.o
	nvcc -lib -o ${BUILD}/ordlink.a ${BUILD}/kernel_roi_align.cu.o ${BUILD}/kernel_roi_align_grad.cu.o ${BUILD}/roi_align.o ${BUILD}/devicelink.o

# D
${BUILD}/libroi_align_cpu_cuda.so: ${CCSRC}/kernel_roi_align.cc ${CCSRC}/kernel_roi_align.h ${BUILD}/kernel_roi_align.cu.o ${CCSRC}/kernel_roi_align_grad.cc ${CCSRC}/kernel_roi_align_grad.h ${BUILD}/kernel_roi_align_grad.cu.o ${BUILD}/devicelink.o ${BUILD}/ordlink.a
	g++ -shared -o ${BUILD}/libroi_align_cpu_cuda.so \
	${CCSRC}/kernel_roi_align.cc \
	${CCSRC}/kernel_roi_align_grad.cc \
	${BUILD}/ordlink.a \
	${TF_CFLAGS} \
	-fPIC -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart ${TF_LFLAGS} -D GOOGLE_CUDA=1

# 2. Compilation without CUDA toolkit

# Compiles the CPU kernels only if no cuda toolkit is available
${BUILD}/roi_align_cpu.o: ${CCSRC}/roi_align.cc
	g++ -c -o ${BUILD}/roi_align_cpu.o ${CCSRC}/roi_align.cc \
	${TF_CFLAGS} -fPIC

${BUILD}/libroi_align_cpu.so: ${CCSRC}/kernel_roi_align.cc ${BUILD}/roi_align_cpu.o
	g++ -shared -o ${BUILD}/libroi_align_cpu.so \
	${CCSRC}/kernel_roi_align.cc \
	${BUILD}/roi_align_cpu.o \
	${TF_CFLAGS} -fPIC ${TF_LFLAGS}

devicelink: ${BUILD}/devicelink.o

ordlink: ${BUILD}/ordlink.a

no_cuda: ${BUILD}/libroi_align_cpu.so

clean:
	rm -f ${BUILD}/ordlink.a
	rm -f ${BUILD}/kernel_roi_align.cu.o
	rm -f ${BUILD}/kernel_roi_align_grad.cu.o
	rm -f ${BUILD}/libroi_align_cpu_cuda.so
	rm -f ${BUILD}/libroi_align_cpu.so
	rm -f ${BUILD}/roi_align.o
	rm -f ${BUILD}/roi_align_cpu.o
	rm -f ${BUILD}/devicelink.o
