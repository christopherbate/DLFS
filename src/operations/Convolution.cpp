#include "Convolution.hpp"
#include "GPU.hpp"
#include "Logging.hpp"

#include <cassert>

using namespace DLFS;
using namespace std;

Convolution::Convolution()
    : m_features(nullptr),
      m_filter(nullptr),
      m_bias(nullptr),
      m_output(nullptr)
{
    Reset();
    m_strides = {1, 1};
    m_padding = {1, 1};    
}

Convolution::Convolution(std::array<int, 2> padding, std::array<int, 2> stride)
    : m_features(nullptr),
      m_filter(nullptr),
      m_bias(nullptr),
      m_output(nullptr)
{
    Reset();
    m_strides = stride;
    m_padding = padding;    
}

void Convolution::Reset()
{
    checkCudaErrors(cudnnCreateConvolutionDescriptor(&m_convDesc));
    m_convFwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    m_scaling = {1.0, 0.0};
    m_workspaceSize = 0;
    m_workspaceBuffer = NULL;
}

Convolution::~Convolution()
{
    checkCudaErrors(cudnnDestroyConvolutionDescriptor(m_convDesc));
}

TensorShape Convolution::Prepare()
{
    assert(m_features != nullptr);
    assert(m_filter != nullptr);

    checkCudaErrors(cudnnSetConvolution2dDescriptor(m_convDesc, m_padding[0], m_padding[1],
                                                    m_strides[0], m_strides[1], 1, 1,
                                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    cudnnSetConvolutionMathType(m_convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

    TensorShape outputDims;
    checkCudaErrors(cudnnGetConvolution2dForwardOutputDim(m_convDesc, m_features->GetTensorDesc(),
                                                          m_filter->GetFilterDesc(), &outputDims[0],
                                                          &outputDims[1], &outputDims[2], &outputDims[3]));
    return outputDims;
}

void Convolution::Execute()
{
    assert(m_output != nullptr);
    assert(m_features->GetPointer() != nullptr);
    assert(m_filter->GetPointer() != nullptr);    
    assert(m_workspaceBuffer == NULL);
    assert(m_workspaceSize == 0);
    assert(m_output->GetPointer() != NULL);

    size_t wsSize = 0;

    // TODO: this should only be done when dimensions / algorithm change.
    cudnnGetConvolutionForwardWorkspaceSize(GPUContext.GetCUDNNHandle(),
                                            m_features->GetTensorDesc(), m_filter->GetFilterDesc(),
                                            m_convDesc, m_output->GetTensorDesc(), m_convFwdAlg, &wsSize);

    cout << "Alg PRECOMP GEMM needs " << wsSize << " bytes of GPU workspace." << endl;

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
}
