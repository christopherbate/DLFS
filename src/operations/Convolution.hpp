#pragma once

#include "tensor/AutoDiff.hpp"
#include "tensor/Tensor.hpp"

#include <array>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

namespace DLFS
{

template<typename T>
class Convolution : public TrackableOp
{
public:
    Convolution();
    Convolution(std::array<int, 2> padding, std::array<int, 2> stride);
    ~Convolution();

    TensorShape Prepare();

    void Execute();

    inline void SetFilter(TensorPtr<T> p)
    {
        m_filter = p;
    }

    inline void SetFeatures(TensorPtr<T> p)
    {
        m_features = p;
    }

    inline void SetOutput(TensorPtr<T> p)
    {
        m_output = p;
    }

private:
    void Reset();

    TensorPtr<T> m_features;
    TensorPtr<T> m_filter;
    TensorPtr<T> m_bias;
    TensorPtr<T> m_output;

    cudnnConvolutionDescriptor_t m_convDesc;
    cudnnConvolutionFwdAlgo_t m_convFwdAlg;

    uint8_t *m_workspaceBuffer;
    size_t m_workspaceSize;

    std::array<float, 2> m_scaling;
    std::array<int, 2> m_strides;
    std::array<int, 2> m_padding;
};

} // namespace DLFS
