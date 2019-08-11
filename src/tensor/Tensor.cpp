#include "Tensor.hpp"
#include "AutoDiff.hpp"

#include <iostream>
#include <memory>
#include <vector>

#include "operations/Convolution.hpp"

using namespace DLFS;
using namespace std;

template <typename T>
Tensor<T>::Tensor() :
    m_pitch{sizeof(T)}
{
    m_bufferSize = 0;
    m_shape = {0, 0, 0, 0};
    m_deviceBuffer = NULL;    
    m_dataType = CUDNN_DATA_FLOAT;
    m_filterDesc = NULL;
    m_tensorDesc = NULL;
}

template <typename T>
Tensor<T>::~Tensor()
{
    Deallocate();
    if (m_filterDesc)
        cudnnDestroyFilterDescriptor(m_filterDesc);
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
    size_t needed_bytes = GetExpectedSize();
    if (needed_bytes > m_bufferSize || m_deviceBuffer == NULL)
    {
        m_bufferSize = needed_bytes;
        Allocate();
    }
}

template <typename T>
void Tensor<T>::Allocate()
{
    if (m_deviceBuffer != NULL)
    {
        Deallocate();
    }
    cout << "Allocating tensor " << m_name << ":" << m_id << " " << (float)m_bufferSize/1024000.0 << " Mb"<< std::endl;
    checkCudaErrors(cudaMalloc(&m_deviceBuffer, m_bufferSize));
}

template <typename T>
void Tensor<T>::Deallocate()
{
    if (m_deviceBuffer != NULL)
    {
        cout << "Attempting to deallocate Tensor " << m_name << ":" << m_id << endl;
        checkCudaErrors(cudaFree(m_deviceBuffer));
        m_deviceBuffer = NULL;
    }
}

template <typename T>
TensorPtr<T> Tensor<T>::Convolve(TensorPtr<T> filter)
{
    shared_ptr<Convolution<T>> convOp = make_shared<Convolution<T>>();
    convOp->SetFilter(filter);
    convOp->SetFeatures(this->shared_from_this());

    // Create the output tensor.
    TensorShape outputShape = convOp->Prepare();
    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();    
    outputTensor->SetName("ConvOutput");
    outputTensor->SetShape(outputShape);
    outputTensor->AllocateIfNecessary();
    
    convOp->SetOutput(outputTensor);

    cout << "Conv op prepared, output shape: " << outputTensor->PrintShape() <<  endl;

    convOp->Execute();

    cout << "Executed " << endl;

    ADContext.AddOp(convOp);

    return outputTensor;
}

template class Tensor<float>;
template class Tensor<uint8_t>;