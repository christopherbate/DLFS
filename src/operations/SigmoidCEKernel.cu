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
};

void SetTensorInfo(TensorInfoCUDA &ti, TensorShape &shape) {
    ti.n = shape[0];
    ti.h = shape[1];
    ti.w = shape[2];
    ti.c = shape[3];
}

/**
 * Declarations
 */
__global__ void
SigmoidCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint16_t *labels,
                         TensorInfoCUDA outputShape, float *output);

__global__ void
SigmoidCrossEntropyBackwardFloat(TensorInfoCUDA logitShape, float *logits,
                                 TensorInfoCUDA labelShape, uint16_t *labels,
                                 TensorInfoCUDA outputShape, float *output);

/**
 * Kernels
 */

extern "C" void LaunchSigmoidCEKernel(CustomOpDataType dataType,
                                      TensorShape logitShape, void *logits,
                                      TensorShape labelShape, void *labels,
                                      TensorShape outputShape, void *output) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3];
    SetTensorInfo(ti[0], logitShape);
    SetTensorInfo(ti[1], labelShape);
    SetTensorInfo(ti[2], outputShape);

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.INFO() << "Launching sigmoid cross entropy (float) kernel with "
                   << threads << " threads";
        SigmoidCrossEntropyFloat<<<1, threads>>>(ti[0], (float *)logits, ti[1],
                                                 (uint16_t *)labels, ti[2],
                                                 (float *)output);
        break;
    default:
        throw std::runtime_error("Not implemented.");
    }
}

extern "C" void
LaunchSigmoidCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape,
                              void *logits, TensorShape labelShape,
                              void *labels, TensorShape outputShape,
                              void *output) {
    int threads = logitShape[0];

    TensorInfoCUDA ti[3];
    SetTensorInfo(ti[0], logitShape);
    SetTensorInfo(ti[1], labelShape);
    SetTensorInfo(ti[2], outputShape);

    switch (dataType) {
    case CustomOpDataType::Float:
        LOG.INFO()
            << "Launching sigmoid cross entropy backward (float) kernel with "
            << threads << " threads";
        SigmoidCrossEntropyBackwardFloat<<<1, threads>>>(
            ti[0], (float *)logits, ti[1], (uint16_t *)labels, ti[2],
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
SigmoidCrossEntropyFloat(TensorInfoCUDA logitShape, float *logits,
                         TensorInfoCUDA labelShape, uint16_t *labels,
                         TensorInfoCUDA outputShape, float *output) {
    unsigned int batchIdx = threadIdx.x;

    // Check to make sure we are not out of bounds.
    if (batchIdx < logitShape.n) {
        uint16_t label = labels[batchIdx];
        for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {
            // This is taken from tensor flow docs.
            // Calculates sigmoid CE with good numerical stability for
            // negative x
            unsigned int index = batchIdx * logitShape.c + classIdx;
            float x = logits[index];

            float z = label == classIdx ? 1.0f : 0.0f;
            float ce = max(x, 0.0) - x * z + logf(1.0f + expf(-abs(x)));

            // No reduction case:
            // Index is channelSize*batchIdx + classIdx.
            output[index] = ce;
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
                                 TensorInfoCUDA labelShape, uint16_t *labels,
                                 TensorInfoCUDA outputShape, float *output) {
    unsigned int batchIdx = threadIdx.x;

    // Check to make sure we are not out of bounds.
    if (batchIdx < logitShape.n) {
        uint16_t label = labels[batchIdx];
        for (unsigned int classIdx = 0; classIdx < logitShape.c; classIdx++) {
            // This is taken from tensor flow docs.
            // Calculates sigmoid CE with good numerical stability for
            // negative x
            unsigned int index = batchIdx * logitShape.c + classIdx;
            float x = logits[index];

            float z = label == classIdx ? 1.0f : 0.0f;
            float firstTerm = x > 0.0f ? 1.0f : 0.0f;
            float expAbs = expf(-abs(x));
            float logTerm = expAbs * (1.0f / (1 + expAbs));
            float dCEdX = firstTerm - z - logTerm;

            // No reduction case:
            // Index is channelSize*batchIdx + classIdx.
            output[index] = dCEdX;
        }
    }
}