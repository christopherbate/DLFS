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
    m_convBwdFilterAlg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    m_convBwdDataAlg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
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
    assert(m_features->GetDevicePointer() != nullptr);
    assert(m_filter->GetDevicePointer() != nullptr);
    assert(m_output->GetDevicePointer() != nullptr);

    size_t wsSize = 0;

    // TODO: this should only be done when dimensions / algorithm change.
    cudnnGetConvolutionForwardWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                            m_features->GetTensorDesc(),
                                            m_filter->GetFilterDesc(),
                                            m_convDesc, m_output->GetTensorDesc(), m_convFwdAlg, &wsSize);

    LOG.INFO() << "Executing cudnn ConvFwd kernel, " << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

    unsigned char *devWs = NULL;
    if (wsSize > 0)
        checkCudaErrors(cudaMalloc(&devWs, wsSize));

    checkCudaErrors(cudnnConvolutionForward(GPUContext.GetCUDNNHandle(), &m_scaling[0],
                                            m_features->GetTensorDesc(), m_features->GetDevicePointer(),
                                            m_filter->GetFilterDesc(), m_filter->GetDevicePointer(),
                                            m_convDesc, m_convFwdAlg, devWs,
                                            wsSize, &m_scaling[1],
                                            m_output->GetTensorDesc(), m_output->GetDevicePointer()));
    if (wsSize > 0)
        checkCudaErrors(cudaFree(devWs));

    cudaDeviceSynchronize();    
}

template <typename T>
void Convolution<T>::ExecuteBackward()
{
    assert(m_features->GetDevicePointer() != nullptr);
    assert(m_filter->GetDevicePointer() != nullptr);

    size_t wsSize = 0;
    unsigned char *devWs = NULL;

    m_features->IncrementBackwardPass();

    T blendFactors[2] = {1, 1};

    // Get grad with respect to filter.
    if (m_filter->GetGradFlag())
    {        
        blendFactors[1] = m_filter->IsFirstBackwardPass() ? 0 : 1;        
        m_filter->IncrementBackwardPass();

        // TODO: this should only be done when dimensions / algorithm change.
        cudnnGetConvolutionBackwardFilterWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                                       m_features->GetTensorDesc(),
                                                       m_output->GetTensorDesc(),
                                                       m_convDesc,
                                                       m_filter->GetGradFilterDesc(),
                                                       m_convBwdFilterAlg,
                                                       &wsSize);

        LOG.INFO() << "Execuing cudnn ConvBwdFilter kernel " << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

        if (wsSize > 0)
            checkCudaErrors(cudaMalloc(&devWs, wsSize));

        checkCudaErrors(cudnnConvolutionBackwardFilter(GPUContext.GetCUDNNHandle(), &blendFactors[0],
                                                       m_features->GetTensorDesc(), m_features->GetDevicePointer(),
                                                       m_output->GetTensorDesc(),
                                                       m_output->GetGradPointer(),
                                                       m_convDesc, m_convBwdFilterAlg, devWs,
                                                       wsSize, &blendFactors[1],
                                                       m_filter->GetFilterDesc(),
                                                       m_filter->GetGradPointer()));
    }

    if (m_features->GetGradFlag())
    {
        size_t bwdDataWsSize = 0;        

        blendFactors[1] = m_features->IsFirstBackwardPass() ? 0 : 1;        
        m_features->IncrementBackwardPass();

        checkCudaErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                                                     m_filter->GetFilterDesc(), m_output->GetTensorDesc(),
                                                                     m_convDesc, m_features->GetTensorDesc(),
                                                                     m_convBwdDataAlg, &bwdDataWsSize));

        if (bwdDataWsSize > wsSize)
        {
            checkCudaErrors(cudaFree(devWs));
            checkCudaErrors(cudaMalloc(&devWs, bwdDataWsSize));
            wsSize = bwdDataWsSize;
        }

        LOG.INFO() << "Execuing cudnn ConvBwdData kernel " << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

        checkCudaErrors(cudnnConvolutionBackwardData(GPUContext.GetCUDNNHandle(), &blendFactors[0],
                                                     m_filter->GetFilterDesc(), m_filter->GetDevicePointer(),
                                                     m_output->GetTensorDesc(), m_output->GetGradPointer(),
                                                     m_convDesc, m_convBwdDataAlg,
                                                     devWs, wsSize, &blendFactors[1], m_features->GetTensorDesc(),
                                                     m_features->GetGradPointer()));
    }

    if (wsSize > 0)
    {
        checkCudaErrors(cudaFree(devWs));
    }

    cudaDeviceSynchronize();    
}

template class Convolution<float>;
template class Convolution<uint8_t>;
template class Convolution<uint16_t>;