#include <cuda_runtime.h>
#include <iostream>

#include "tensor/Tensor.hpp"
#include "Logging.hpp"

using namespace DLFS;
using namespace std;

__global__ void
PointwisePowerFloat(const float *A, const float power, float *C, int numElements);

extern "C" void LaunchPowerKernel(CustomOpDataType dataType,
                        void *inputBuffer, int linearLength,
                        void *power, void* outputBuffer)
{
    int tensorSize = linearLength;
    int thrPerBlock = 256;
    int blocksPerGrid = (tensorSize + thrPerBlock - 1) / thrPerBlock;

    switch(dataType){
        case CustomOpDataType::Float:
            LOG.INFO() << "Launching PointwisePower (float) kernel";
            PointwisePowerFloat<<<blocksPerGrid, thrPerBlock>>>((const float*)inputBuffer,
                                                        *(const float *)power, 
                                                        (float*)outputBuffer,
                                                        tensorSize);
            break;                                                
        default:
            throw std::runtime_error("Not implemented.");
    }    
}

/* Pointwise power kernel */
__global__ void
PointwisePowerFloat(const float *A, const float power, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = powf(A[0], power);
    }
}