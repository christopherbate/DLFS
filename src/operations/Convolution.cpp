#include "Convolution.hpp"
#include "GPU.hpp"
#include "Logging.hpp"

#include <cassert>

using namespace DLFS;
using namespace std;

template <typename T>
Convolution<T>::Convolution()
    : m_features(nullptr),
      m_filter(nullptr),
      m_bias(nullptr),
      m_output(nullptr)
{
    Reset();
    m_strides = {1, 1};
    m_padding = {1, 1};
}

template <typename T>
Convolution<T>::Convolution(std::array<int, 2> padding,
                            std::array<int, 2> stride)
    : m_features(nullptr),
      m_filter(nullptr),
      m_bias(nullptr),
      m_output(nullptr)
{
    Reset();
    m_strides = stride;
    m_padding = padding;
}

template <typename T>
void Convolution<T>::Reset()
{
    checkCudaErrors(cudnnCreateConvolutionDescriptor(&m_convDesc));
    m_convFwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    m_convBwdAlg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    m_scaling = {1.0, 0.0};
    m_workspaceSize = 0;
    m_workspaceBuffer = NULL;
    SetName("ConvOp");
}

template <typename T>
Convolution<T>::~Convolution()
{
    checkCudaErrors(cudnnDestroyConvolutionDescriptor(m_convDesc));
}

template <typename T>
TensorShape Convolution<T>::Prepare()
{
    assert(m_features != nullptr);
    assert(m_filter != nullptr);

    checkCudaErrors(cudnnSetConvolution2dDescriptor(m_convDesc, m_padding[0], m_padding[1],
                                                    m_strides[0], m_strides[1], 1, 1,
                                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    cudnnSetConvolutionMathType(m_convDesc, CUDNN_DEFAULT_MATH);

    TensorShape outputDims;
    checkCudaErrors(cudnnGetConvolution2dForwardOutputDim(m_convDesc, m_features->GetTensorDesc(),
                                                          m_filter->GetFilterDesc(), &outputDims[0],
                                                          &outputDims[3], &outputDims[1], &outputDims[2]));
    return outputDims;
}

template <typename T>
void Convolution<T>::ExecuteForward()
{
    assert(m_output != nullptr);
    assert(m_workspaceBuffer == NULL);
    assert(m_workspaceSize == 0);
    assert(m_features->GetPointer() != nullptr);
    assert(m_filter->GetPointer() != nullptr);
    assert(m_output->GetPointer() != nullptr);

    size_t wsSize = 0;

    // TODO: this should only be done when dimensions / algorithm change.
    cudnnGetConvolutionForwardWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                            m_features->GetTensorDesc(),
                                            m_filter->GetFilterDesc(),
                                            m_convDesc, m_output->GetTensorDesc(), m_convFwdAlg, &wsSize);

    cout << "Alg PRECOMP GEMM needs " << (float)wsSize / 1024000.0 << " Mb of GPU workspace." << endl;

    unsigned char *devWs = NULL;
    if (wsSize > 0)
        checkCudaErrors(cudaMalloc(&devWs, wsSize));

    checkCudaErrors(cudnnConvolutionForward(GPUContext.GetCUDNNHandle(), &m_scaling[0],
                                            m_features->GetTensorDesc(), m_features->GetPointer(),
                                            m_filter->GetFilterDesc(), m_filter->GetPointer(),
                                            m_convDesc, m_convFwdAlg, devWs,
                                            wsSize, &m_scaling[1],
                                            m_output->GetTensorDesc(), m_output->GetPointer()));
    if (wsSize > 0)
        checkCudaErrors(cudaFree(devWs));

    cudaDeviceSynchronize();
    std::cout << "Convolution executed." << std::endl;
}

template <typename T>
void Convolution<T>::ExecuteBackward(TensorPtr<T> dy)
{
    assert(m_features->GetPointer() != nullptr);
    assert(m_filter->GetPointer() != nullptr);

    size_t wsSize = 0;

    // TODO: this should only be done when dimensions / algorithm change.
    cudnnGetConvolutionBackwardFilterWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                                   m_features->GetTensorDesc(),
                                                   dy->GetTensorDesc(),
                                                   m_convDesc,
                                                   m_filter->GetGradFilterDesc(),
                                                   m_convBwdAlg,
                                                   &wsSize);

    cout << "Alg BWD FILTER ALGO 0 needs " << (float)wsSize / 1024000.0 << " Mb of GPU workspace." << endl;

    unsigned char *devWs = NULL;

    if (wsSize > 0)
        checkCudaErrors(cudaMalloc(&devWs, wsSize));

    checkCudaErrors(cudnnConvolutionBackwardFilter(GPUContext.GetCUDNNHandle(), &m_scaling[0],
                                                   m_features->GetTensorDesc(), m_features->GetPointer(),
                                                   dy->GetTensorDesc(),
                                                   dy->GetPointer(),
                                                   m_convDesc, m_convBwdAlg, devWs,
                                                   wsSize, &m_scaling[1],
                                                   m_filter->GetGradFilterDesc(),
                                                   m_filter->GetGradPointer()));
    if (wsSize > 0)
        checkCudaErrors(cudaFree(devWs));

    cudaDeviceSynchronize();
    std::cout << "Convolution_grad_filter executed." << std::endl;
}

template class Convolution<float>;
template class Convolution<uint8_t>;