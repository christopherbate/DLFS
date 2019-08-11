#pragma once

#include "tensor/AutoDiff.hpp"
#include "tensor/Tensor.hpp"

#include <array>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

namespace DLFS
{

class Convolution : public TrackableOp
{
public:
    Convolution();
    Convolution(std::array<int, 2> padding, std::array<int, 2> stride);
    ~Convolution();

    TensorShape Prepare();

    void Execute();

    inline void SetFilter(TensorPtr p)
    {
        m_filter = p;
    }

    inline void SetFeatures(TensorPtr p)
    {
        m_features = p;
    }

    inline void SetOutput(TensorPtr p)
    {
        m_output = p;
    }

private:
    void Reset();

    TensorPtr m_features;
    TensorPtr m_filter;
    TensorPtr m_bias;
    TensorPtr m_output;

    cudnnConvolutionDescriptor_t m_convDesc;
    cudnnConvolutionFwdAlgo_t m_convFwdAlg;

    uint8_t *m_workspaceBuffer;
    size_t m_workspaceSize;

    std::array<float, 2> m_scaling;
    std::array<int, 2> m_strides;
    std::array<int, 2> m_padding;
};

} // namespace DLFS
