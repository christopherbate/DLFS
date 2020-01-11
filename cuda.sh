#!/bin/bash
nvcc -I./lib/ --gpu-architecture=sm_75 --lib lib/operations/PointwiseKernels.cu lib/operations/SoftmaxCEKernel.cu lib/operations/SigmoidCEKernel.cu -o lib/kernels_75.o
nvcc -I./lib/ --gpu-architecture=sm_52 --lib lib/operations/PointwiseKernels.cu lib/operations/SoftmaxCEKernel.cu lib/operations/SigmoidCEKernel.cu -o lib/kernels_52.o
