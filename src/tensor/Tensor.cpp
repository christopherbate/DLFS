#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include "operations/Convolution.hpp"
#include "Tensor.hpp"
#include "AutoDiff.hpp"

using namespace DLFS;
using namespace std;

template <typename T>
Tensor<T>::Tensor()
{
    m_shape = {0, 0, 0, 0};
    m_dataType = CUDNN_DATA_FLOAT;
}

template <typename T>
Tensor<T>::~Tensor()
{
    Deallocate();
    if (m_filterDesc)
        cudnnDestroyFilterDescriptor(m_filterDesc);
    if (m_dwFilterDesc)
        cudnnDestroyFilterDescriptor(m_dwFilterDesc);
    if (m_tensorDesc)
        cudnnDestroyTensorDescriptor(m_tensorDesc);
}

template <typename T>
void Tensor<T>::SetShape(TensorShape shape)
{
    m_isFilter = false;
    m_shape = shape;
    FillCUDNNDesc();
}

template <typename T>
void Tensor<T>::SetShape(int b, int h, int w, int c)
{
    m_isFilter = false;
    m_shape = std::array<int, 4>{b, h, w, c};
    FillCUDNNDesc();
}

/**
 * we use the NHWC filter shape, 
 * which means the filters take the form of 
 * KRSC (output, rows, cols, input)
 * **/
template <typename T>
void Tensor<T>::SetFilterShape(int numInputChn,
                               int numOutputChn,
                               int rows, int cols)
{
    m_isFilter = true;
    m_shape = {numOutputChn, rows, cols, numInputChn};
    FillCUDNNDesc();
}

/**
 * Sets up the tensor descriptor for CUDNN. Must be called after
 * the shape of the tensor changes.
 */
template <typename T>
void Tensor<T>::FillCUDNNDesc()
{
    if (m_isFilter)
    {
        if (!m_filterDesc)
            checkCudaErrors(cudnnCreateFilterDescriptor(&m_filterDesc));
        checkCudaErrors(cudnnSetFilter4dDescriptor(m_filterDesc, m_dataType,
                                                   CUDNN_TENSOR_NHWC, m_shape[0], m_shape[3],
                                                   m_shape[1], m_shape[2]));

        if (m_calcGrad)
        {
            if (!m_dwFilterDesc)
                checkCudaErrors(cudnnCreateFilterDescriptor(&m_dwFilterDesc));
            checkCudaErrors(cudnnSetFilter4dDescriptor(m_dwFilterDesc,
                                                       m_dataType, CUDNN_TENSOR_NHWC,
                                                       m_shape[0], m_shape[3], m_shape[1],
                                                       m_shape[2]));
        }
    }
    else
    {
        if (!m_tensorDesc)
            checkCudaErrors(cudnnCreateTensorDescriptor(&m_tensorDesc));
        checkCudaErrors(cudnnSetTensor4dDescriptor(m_tensorDesc, CUDNN_TENSOR_NHWC, m_dataType,
                                                   m_shape[0], m_shape[3], m_shape[1], m_shape[2]));       
    }
}

template <typename T>
void Tensor<T>::AllocateIfNecessary()
{
    size_t needed_bytes = GetLinearSize() * sizeof(T);
    bool needGradBuffer = m_calcGrad && m_deviceBufferGrad == nullptr;
    if (needed_bytes > m_bufferSize || m_deviceBuffer == nullptr || needGradBuffer)
    {
        m_bufferSize = needed_bytes;
        Allocate();
    }
}

template <typename T>
void Tensor<T>::Allocate()
{
    if (m_deviceBuffer != nullptr || m_deviceBufferGrad != nullptr)
        Deallocate();

    cout << "Allocating tensor " << GetName() << ":" << GetId()
         << " " << (float)m_bufferSize / 1024000.0 << " Mb" << std::endl;
    checkCudaErrors(cudaMalloc(&m_deviceBuffer, m_bufferSize));

    if (m_calcGrad)
    {
        cout << "Allocating grad tensor " << GetName() << ":" << GetId()
             << " " << (float)m_bufferSize / 1024000.0 << " Mb" << std::endl;
        checkCudaErrors(cudaMalloc(&m_deviceBufferGrad, m_bufferSize));
    }
}

template <typename T>
void Tensor<T>::Deallocate()
{    
    if (m_deviceBuffer != nullptr)
    {
        cout << "Attempting to deallocate Tensor " << GetName() << ":" << GetId() << endl;
        checkCudaErrors(cudaFree(m_deviceBuffer));
        if(m_deviceBufferGrad == m_deviceBuffer)
            m_deviceBufferGrad = nullptr;        
        m_deviceBuffer = nullptr;
        m_bufferSize = 0;
    }
    if (m_deviceBufferGrad != nullptr)
    {
        cout << "Attempting to deallocate grad Tensor " << GetName() << ":" << GetId() << endl;
        checkCudaErrors(cudaFree(m_deviceBufferGrad));
        m_deviceBufferGrad = nullptr;
    }
}

template <typename T>
TensorPtr<T> Tensor<T>::Convolve(TensorPtr<T> filter,
                                 Pad2d padding, Stride2d stride)
{
    shared_ptr<Convolution<T>> convOp =
        make_shared<Convolution<T>>(padding, stride);
    convOp->SetFilter(filter);
    convOp->SetFeatures(this->shared_from_this());

    // Create the output tensor.
    TensorShape outputShape = convOp->Prepare();
    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();

    // Put this variable into the active set if either inputs are active
    if (filter->GetGradFlag() || GetGradFlag())
        outputTensor->SetGradFlag(true);

    outputTensor->SetName("ConvOutput");
    outputTensor->SetShape(outputShape);
    outputTensor->AllocateIfNecessary();

    convOp->SetOutput(outputTensor);

    cout << "Conv op prepared, output shape: "
         << outputTensor->PrintShape() << endl;

    convOp->ExecuteForward();

    cout << "Executed " << endl;

    ADContext.AddOp(convOp);

    return outputTensor;
}

template class Tensor<float>;
template class Tensor<uint8_t>;