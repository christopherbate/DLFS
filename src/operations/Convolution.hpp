#pragma once

#include "tensor/AutoDiff.hpp"
#include "tensor/Tensor.hpp"

#include <array>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

#include "operations/OpsCommon.hpp"

namespace DLFS
{

template <typename T>
class Convolution : public TrackableOp
{
public:
    Convolution();
    Convolution(Pad2d padding, Stride2d stride);
    ~Convolution();

    TensorShape Prepare();

    void ExecuteForward();
    void ExecuteBackward();

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

    TensorBasePtr GetOutputTensor()
    {
        return m_output;
    }

private:
    void Reset();

    TensorPtr<T> m_features;
    TensorPtr<T> m_filter;
    TensorPtr<T> m_bias;
    TensorPtr<T> m_output;
    TensorPtr<T> m_dy{nullptr};

    cudnnConvolutionDescriptor_t m_convDesc;

    /**
     * These variables encode the specific
     * algorithm used by cudnn functions.
     * Each algorithm also might induce different kernels
     * depending on the compute capability and size of 
     * the tensors. Right now we have the algs fixed,
     * but in the future these should be selected dynamically.
     */    
    cudnnConvolutionFwdAlgo_t m_convFwdAlg;
    cudnnConvolutionBwdFilterAlgo_t m_convBwdFilterAlg;
    cudnnConvolutionBwdDataAlgo_t m_convBwdDataAlg;

    uint8_t *m_workspaceBuffer;
    size_t m_workspaceSize;

    std::array<float, 2> m_scaling;
    std::array<int, 2> m_strides;
    std::array<int, 2> m_padding;
};

template <typename T>
using ConvOpPtr = std::shared_ptr<Convolution<T>>;

} // namespace DLFS
