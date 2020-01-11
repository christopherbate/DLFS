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
SoftmaxCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint32_t *labels,
                         TensorInfoCUDA outputShape, float *output, bool reduce_mean);

__global__ void
SoftmaxCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits,
                                 TensorInfoCUDA labelShape, uint32_t *labels,
                                 TensorInfoCUDA outputShape, float *output);

/**
 * Kernels
 */

extern "C" void 
LaunchSoftmaxCEKernel(CustomOpDataType dataType,
                                      TensorShape logitShape, void *logits,
                                      TensorShape labelShape, void *labels,
                                      TensorShape outputShape, void *output, bool reduce_mean) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3] = {TensorInfoCUDA(logitShape),
        TensorInfoCUDA(labelShape), TensorInfoCUDA(outputShape)};        

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.DEBUG() << "Launching softmax cross entropy (float) kernel with "
                   << threads << " threads";
        SoftmaxCrossEntropyFloat<<<1, threads, threads>>>(ti[0], (float *)logits, ti[1],
                                                 (uint32_t *)labels, ti[2],
                                                 (float *)output, reduce_mean);
        break;
    default:
        throw std::runtime_error("Not implemented.");
    }
}

extern "C" void
LaunchSoftmaxCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape,
                              void *logits, TensorShape labelShape,
                              void *labels, TensorShape outputShape,
                              void *output) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3] = {TensorInfoCUDA(logitShape),
        TensorInfoCUDA(labelShape), TensorInfoCUDA(outputShape)};       

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.DEBUG()
            << "Launching softmax cross entropy backward (float) kernel with "
            << threads << " threads";
        SoftmaxCrossEntropyBackwardFloat<<<1, threads>>>(
            ti[0], (float *)logits, ti[1], (uint32_t *)labels, ti[2],
            (float *)output);
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
SoftmaxCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint32_t *labels,
                         TensorInfoCUDA outputShape, float *output, bool reduce_mean) {
    unsigned int batchIdx = threadIdx.x;
    extern __shared__ float sdata[];    

    // Check to make sure we are not out of bounds.
    if (batchIdx > logitShape.n-1) 
        return;

    // float normalization = logitShape.n*logitShape.c;
    float exp_sum = 0.0f;
    float loss = 0.0f;
    
    for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {        
        unsigned int index = batchIdx * logitShape.c + classIdx;            
        exp_sum += expf(logits[index]);                    
    }        
    loss = logf(exp_sum)-logits[batchIdx*logitShape.c+labels[batchIdx]];
    sdata[batchIdx] = loss;    

    if(!reduce_mean){
        output[batchIdx] = sdata[batchIdx];
        return;
    }

    // Parallel reduction - requires batch to be power of 2
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
SoftmaxCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits,
                                 TensorInfoCUDA labelShape, uint32_t *labels,
                                 TensorInfoCUDA outputShape, float *output) {
    unsigned int batchIdx = threadIdx.x;
    extern __shared__ float sdata[];    

    // Check to make sure we are not out of bounds.
    if (batchIdx > logitShape.n-1) 
        return;

    // float normalization = logitShape.n*logitShape.c;
    float exp_sum = 0.0f;
    float loss = 0.0f;
    
    for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {        
        unsigned int index = batchIdx * logitShape.c + classIdx;            
        exp_sum += expf(logits[index]);                    
    }        
    loss = logf(exp_sum)-logits[batchIdx*logitShape.c+labels[batchIdx]];
    output[batchIdx] = loss;      
}