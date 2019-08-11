#include "Tensor.hpp"
#include "AutoDiff.hpp"

#include <iostream>
#include <memory>
#include <vector>

#include "operations/Convolution.hpp"

using namespace DLFS;
using namespace std;

Tensor::Tensor()
{
    m_bufferSize = 0;
    m_shape = {0, 0, 0, 0};
    m_deviceBuffer = NULL;
    m_pitch = 1;
    m_dataType = CUDNN_DATA_FLOAT;
    m_filterDesc = NULL;
    m_tensorDesc = NULL;
}

Tensor::~Tensor()
{
    Deallocate();
    if (m_filterDesc)
        cudnnDestroyFilterDescriptor(m_filterDesc);
    if (m_tensorDesc)
        cudnnDestroyTensorDescriptor(m_tensorDesc);
}

void Tensor::SetShape(TensorShape shape)
{
    m_isFilter = false;
    m_shape = shape;
    FillCUDNNDesc();
}

void Tensor::SetShape(int b, int h, int w, int c)
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
void Tensor::SetFilterShape(int numInputChn,
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
void Tensor::FillCUDNNDesc()
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

void Tensor::AllocateIfNecessary()
{
    size_t needed_bytes = GetExpectedSize();
    if (needed_bytes > m_bufferSize || m_deviceBuffer == NULL)
    {
        m_bufferSize = needed_bytes;
        Allocate();
    }
}

void Tensor::Allocate()
{
    if (m_deviceBuffer != NULL)
    {
        Deallocate();
    }
    checkCudaErrors(cudaMalloc(&m_deviceBuffer, m_bufferSize));
}

void Tensor::Deallocate()
{
    if (m_deviceBuffer != NULL)
    {
        checkCudaErrors(cudaFree(m_deviceBuffer));
        m_deviceBuffer = NULL;
    }
}

TensorPtr Tensor::Convolve(std::shared_ptr<Tensor> filter)
{
    shared_ptr<Convolution> convOp = make_shared<Convolution>();
    convOp->SetFilter(filter);
    convOp->SetFeatures(shared_from_this());

    TensorPtr outputTensor = make_shared<Tensor>();
    const TensorShape &filterShape = filter->GetShape();

    outputTensor->SetShape(m_shape[0], m_shape[1], m_shape[2], filterShape[0]);
    outputTensor->AllocateIfNecessary();

    convOp->SetOutput(outputTensor);

    convOp->Prepare();

    cout << "Conv op prepared." << endl;

    convOp->Execute();

    cout << "Executed " << endl;

    ADContext.AddOp(convOp);

    return outputTensor;
}