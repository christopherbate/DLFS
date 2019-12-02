#include <cuda_runtime.h>
#include <iostream>

#include "Logging.hpp"
#include "tensor/Tensor.hpp"

using namespace DLFS;
using namespace std;

struct TensorInfoCUDA {
    int n;
    int h;
    int w;
    int c;

    TensorInfoCUDA(const TensorShape& shape){
        n = shape[0];
        h = shape[1];
        w = shape[2];
        c = shape[3];
    }
};

/**
 * Declarations
 */
__global__ void
SigmoidCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint32_t *labels,
                         TensorInfoCUDA outputShape, float *output, bool reduce_mean);

__global__ void
SigmoidCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits,
                                 TensorInfoCUDA labelShape, uint32_t *labels,
                                 TensorInfoCUDA outputShape, float *output, bool reduce_mean);

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
        SigmoidCrossEntropyFloat<<<1, threads>>>(ti[0], (float *)logits, ti[1],
                                                 (uint32_t *)labels, ti[2],
                                                 (float *)output, reduce_mean);
        break;
    default:
        throw std::runtime_error("Not implemented.");
    }
}

extern "C" void
LaunchSigmoidCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape,
                              void *logits, TensorShape labelShape,
                              void *labels, TensorShape outputShape,
                              void *output, bool reduce_mean) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3] = {TensorInfoCUDA(logitShape),
        TensorInfoCUDA(labelShape), TensorInfoCUDA(outputShape)};       

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.DEBUG()
            << "Launching sigmoid cross entropy backward (float) kernel with "
            << threads << " threads";
        SigmoidCrossEntropyBackwardFloat<<<1, threads>>>(
            ti[0], (float *)logits, ti[1], (uint32_t *)labels, ti[2],
            (float *)output, reduce_mean);
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

    // Check to make sure we are not out of bounds.
    if (batchIdx < logitShape.n) {
        uint32_t label = labels[batchIdx];
        float normalization = logitShape.n*logitShape.c;
        for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {
            // This is taken from tensor flow docs.
            // Calculates sigmoid CE with good numerical stability for
            // negative x
            unsigned int index = batchIdx * logitShape.c + classIdx;
            float x = logits[index];

            float z = label == classIdx ? x : 0.0f;
            float ce = max(x, 0.0) - z + logf(1.0f + expf(-abs(x)));
            // float ce = z;

            // No reduction case:
            // Index is channelSize*batchIdx + classIdx.
            if (reduce_mean){                
                atomicAdd_block(output, ce/normalization);
            } else {
                output[index] = ce;
            }
        }
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
SigmoidCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits,
                                 TensorInfoCUDA labelShape, uint32_t *labels,
                                 TensorInfoCUDA outputShape, float *output, bool reduce_mean) {
    unsigned int batchIdx = threadIdx.x;

    // Check to make sure we are not out of bounds.
    if (batchIdx < logitShape.n) {
        uint32_t label = labels[batchIdx];
        float normalization = logitShape.n*logitShape.c;
        for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {
            // This is taken from tensor flow docs.
            // Calculates sigmoid CE with good numerical stability for
            // negative x
            
            unsigned int index = batchIdx * logitShape.c + classIdx;
            float x = logits[index];

            float z = label == classIdx ? 1.0f : 0.0f;
            float expAbs = expf(-x);
            float logTerm = expAbs / (1 + expAbs);
            float dCEdX = z - logTerm;

            // No reduction case:
            // Index is channelSize*batchIdx + classIdx.
            if (!reduce_mean){
                output[index] = dCEdX;
            }else{
                output[index] = dCEdX / normalization;
            }            
        }
    }
}