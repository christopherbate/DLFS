#include <iostream>

#include "ConvLayer.hpp"
#include "../Logging.hpp"

namespace DLFS{

/**
 * ConvLayer
 */
template <typename T>
ConvLayer<T>::ConvLayer(cudnnHandle_t &handle, Layer<T> *prevLayer) : Layer<T>(handle, prevLayer)
{
    checkCudaErrors(cudnnCreateFilterDescriptor(&m_filterTd));
    checkCudaErrors(cudnnCreateConvolutionDescriptor(&m_convDesc));
    checkCudaErrors(cudnnSetConvolution2dDescriptor(m_convDesc, 1/*pad h*/,
        1/*pad w*/, 1 /*stride h*/, 1 /*stride w*/, 1 /*dilation h*/, 1 /*dilation w*/,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // Allow conversion for tensor ops where available.
    checkCudaErrors( cudnnSetConvolutionMathType(m_convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );

    m_workspaceBuffer = NULL;
    m_workspaceSize = 0;
    m_filterBuffer = NULL;
}

template <typename T>
ConvLayer<T>::~ConvLayer()
{
    checkCudaErrors(cudnnDestroyFilterDescriptor(m_filterTd));
    checkCudaErrors(cudnnDestroyConvolutionDescriptor(m_convDesc));  

    if(m_workspaceBuffer){
        checkCudaErrors(cudaFree(m_workspaceBuffer));
    }
    if(m_filterBuffer){
        checkCudaErrors(cudaFree(m_filterBuffer));
    }
}

template <typename T>
void ConvLayer<T>::AllocateBuffers()
{
    checkCudaErrors(cudaMalloc( &this->m_outputBuffer, this->m_outputDims.Length()*sizeof(T)));
    checkCudaErrors(cudaMalloc( &m_filterBuffer, m_filterDims.Length()*sizeof(T)));    
    checkCudaErrors(cudaMalloc( &m_workspaceBuffer, m_workspaceSize));    
}

template <typename T>
void ConvLayer<T>::Forward()
{
    float alpha = 1.0, beta = 0.0;
    checkCudaErrors(cudnnConvolutionForward(this->m_handle, (void*)&alpha, this->m_prevLayer->GetOutputTensorDesc(),
        this->m_prevLayer->GetOutputBuffer(), m_filterTd, m_filterBuffer, m_convDesc, m_convAlg, m_workspaceBuffer, m_workspaceSize, 
        (void*)&beta, this->m_outputTd, (void*)this->m_outputBuffer));
}

template <typename T>
void ConvLayer<T>::SetFilerDim( FilterDims &dims )
{
    if(!this->m_prevLayer)
    {
        throw std::runtime_error("ConvLayer has not previous layer.");
    }

    m_filterDims = dims;
    checkCudaErrors( cudnnSetFilter4dDescriptor(m_filterTd, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, /*output feature maps*/ dims.outputFeatures,
        /*input feature maps*/this->m_inputDims.channels, 
        /*filter height*/ dims.height, /*filter width*/dims.width));            

    int b, c, h, w;
    checkCudaErrors( cudnnGetConvolution2dForwardOutputDim(m_convDesc, this->m_prevLayer->GetOutputTensorDesc(),
        m_filterTd, &b, &c, &h, &w));

    this->m_outputDims = TensorDims(b, h, w, c);

    checkCudaErrors( cudnnSetTensor4dDescriptor( this->m_outputTd, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, b, c, h, w));    
}

template <typename T>
void ConvLayer<T>::FindBestAlgorithm()
{
    using namespace std;

    // Allocate the filter weights.
    size_t filterSize = m_filterDims.Length()*sizeof(T);
    uint8_t *filterMem = nullptr;
    checkCudaErrors(cudaMalloc(&filterMem, filterSize));

    // Allocate and initialize tensors (again, only the input tensor is shown):
    size_t inputSize = this->m_inputDims.Length()*sizeof(T);
    uint8_t *inputMem = nullptr;
    checkCudaErrors(cudaMalloc(&inputMem, inputSize));

    // Allocate output size
    size_t outputSize = this->m_outputDims.Length()*sizeof(T);
    uint8_t *outputMem = nullptr;
    checkCudaErrors(cudaMalloc(&outputMem, outputSize));

    checkCudaErrors(cudaDeviceSynchronize());

    // Find the best algorithm.
    int algCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    checkCudaErrors(cudnnFindConvolutionForwardAlgorithm(this->m_handle, 
        this->m_prevLayer->GetOutputTensorDesc(), m_filterTd, m_convDesc, this->m_outputTd, 8, &algCount, perfResults));

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(filterMem));
    checkCudaErrors(cudaFree(inputMem));
    checkCudaErrors(cudaFree(outputMem));

    cout << "Performance Results for this Layer: "<< endl;
    for(int i =0; i < 8; i++)    
    {
        cout << i << ": " << GetConvAlgorithmString(perfResults[i].algo) << ", MathType: " << perfResults[i].mathType <<"\n";
        cout << "  Mem:" << perfResults[i].memory << ", " << " Time: " <<perfResults[i].time << endl;
    }

    m_workspaceSize = perfResults[0].memory;
    m_convAlg = perfResults[0].algo;
    checkCudaErrors(cudaMalloc(&m_workspaceBuffer, m_workspaceSize));
}

template <typename T>
size_t ConvLayer<T>::GetAlgWorkspaceNeeded(cudnnConvolutionFwdAlgo_t alg)
{
    size_t workspaceSize;    
    cudnnStatus_t status;
    status = cudnnGetConvolutionForwardWorkspaceSize(this->m_handle,
            this->m_prevLayer->GetOutputTensorDesc(), m_filterTd, m_convDesc, this->m_outputTd, alg, &workspaceSize);

    if(status != CUDNN_STATUS_SUCCESS){
        throw std::runtime_error( cudaGetErrorName(status) );
    }                   

    return workspaceSize;                                     
}

template <typename T>
void ConvLayer<T>::PrintAllWorkspaces()
{   
    using namespace std;             
    cout << "Workspaces Needed by Algorithm for this ConvLayer: \n";    
    for(int i = 0; i < 8; i++)
    {                
        try {
            size_t ws =  GetAlgWorkspaceNeeded((cudnnConvolutionFwdAlgo_t)i);
            cout << "Alg "<< GetConvAlgorithmString((cudnnConvolutionFwdAlgo_t)i) << ": " << (float)ws/1e6 << " MB\n";            
        } catch (std::exception &e)
        {
            cout << "Alg "<<  GetConvAlgorithmString((cudnnConvolutionFwdAlgo_t)i) << ": UNSUPPORTED - " << e.what() <<  "\n";
        }            
    }    
    cout << endl;
}

template <typename T>
const char* ConvLayer<T>::GetConvAlgorithmString( cudnnConvolutionFwdAlgo_t alg)
{            
    switch(alg){
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "IMPLICIT GEMM";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "IMPLICIT PRECOMP GEMM";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "GEMM";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "DIRECT";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "FFT";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "FFT TILING";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "WINOGRAD";
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD NONFUSED";            
            break;
        default:            
            break;
    }
    return "UNKNOWN";
}

}

template class DLFS::ConvLayer<float>;