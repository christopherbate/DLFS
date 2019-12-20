#include <cuda_runtime.h>
#include <iostream>

#include "Logging.hpp"
#include "tensor/Tensor.hpp"

using namespace DLFS;
using namespace std;

/**
 * Declarations
 */
__global__ void
SigmoidCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint32_t *labels,
                         TensorInfoCUDA outputShape, float *output, bool reduce_mean);

__global__ void
SigmoidCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits, float *dLogits,
                                TensorInfoCUDA labelShape, uint32_t *labels);                    

/**
 * Kernels
 */

extern "C" void 
LaunchSigmoidCEKernel(CustomOpDataType dataType,
                                      TensorShape logitShape, void *logits,
                                      TensorShape labelShape, void *labels,
                                      TensorShape outputShape, void *output, bool reduce_mean) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3] = {TensorInfoCUDA(logitShape),
        TensorInfoCUDA(labelShape), TensorInfoCUDA(outputShape)};        

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.DEBUG() << "Launching sigmoid cross entropy (float) kernel with "
                   << threads << " threads";
        SigmoidCrossEntropyFloat<<<1, threads, logitShape[0]*logitShape[3]>>>(ti[0], (float *)logits, ti[1],
                                                 (uint32_t *)labels, ti[2],
                                                 (float *)output, reduce_mean);
        break;
    default:
        throw std::runtime_error("Not implemented.");
    }
}

extern "C" void
LaunchSigmoidCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape, 
                              void *logits, void *dLogits, TensorShape labelShape,
                              void *labels) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[2] = {TensorInfoCUDA(logitShape),
        TensorInfoCUDA(labelShape)};

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.DEBUG()
            << "Launching sigmoid cross entropy backward (float) kernel with "
            << threads << " threads";
        SigmoidCrossEntropyBackwardFloat<<<1, threads>>>(
            ti[0], (float *)logits, (float*)dLogits, ti[1], (uint32_t *)labels);
        break;
    default:
        throw std::runtime_error("Not implemented.");
    }
}

/**
 * SigmoidCrossEntropyFloat
 * Logits - of shape batch_size x 1 x 1 x num_classes
 * labels - of shape batch_size x 1 x 1 x 1
 *
 * Parallelized over the batch dimension.
 */
__global__ void
SigmoidCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint32_t *labels,
                         TensorInfoCUDA outputShape, float *output, bool reduce_mean) {
    unsigned int batchIdx = threadIdx.x;
    extern __shared__ float sdata[];

    // Check to make sure we are not out of bounds.
    if (batchIdx > logitShape.n-1) 
        return;

    uint32_t label = labels[batchIdx];
    float loss = 0.0;
    for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {            
        unsigned int index = batchIdx * logitShape.c + classIdx;
        float x = logits[index];            
        loss += max(x, 0.0) + logf(1.0f + expf(-abs(x)));                            
    }
    sdata[batchIdx] = loss - logits[batchIdx*logitShape.c + label];        

    if(!reduce_mean){
        output[batchIdx] = sdata[batchIdx];
        return;
    }

    __syncthreads();
    for(unsigned int stride = 1; stride < logitShape.n; stride*=2){
        if((threadIdx.x %(stride*2))==0){
            sdata[threadIdx.x] += sdata[threadIdx.x+stride];
        }
        __syncthreads(); // Sync must happen at every level of the pyramid;
    }

    if (threadIdx.x == 0){
        output[0] = sdata[0] / static_cast<float>(logitShape.n);
    }
}

/**
 * SigmoidCrossEntropyBackwardFloat
 * Logits - of shape batch_size x 1 x 1 x num_classes
 * labels - of shape batch_size x 1 x 1 x 1
 *
 * Parallelized over the batch dimension.
 */
__global__ void
SigmoidCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits, float *dLogits,
                                 TensorInfoCUDA labelShape, uint32_t *labels)
{                                 
    unsigned int batchIdx = threadIdx.x;

    // Check to make sure we are not out of bounds.
    if (batchIdx > logitShape.n-1)
        return;
        
    uint32_t label = labels[batchIdx];
    float normalization = logitShape.n*logitShape.c;
    for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {

        
        unsigned int index = batchIdx * logitShape.c + classIdx;
        float x = logits[index];

        float z = label == classIdx ? 1.0f : 0.0f;
        float expAbs = expf(-x);
        float logTerm = expAbs / (1 + expAbs);
        float dCEdX = z - logTerm;               
        dLogits[index] = dCEdX;
    }    
}