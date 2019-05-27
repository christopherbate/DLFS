#include "InputLayer.hpp"
#include "../Logging.hpp"

namespace DLFS
{

/**
 * InputLayer
 **/
template <typename T>
InputLayer<T>::InputLayer(cudnnHandle_t &handle, Layer<T> *prevLayer) : Layer<T>(handle, NULL)
{
}

template <typename T>
InputLayer<T>::~InputLayer()
{
}

template <typename T>
void InputLayer<T>::SetInputDim(TensorDims &dims)
{
    this->m_inputDims = dims;
    this->m_outputDims = dims;
    checkCudaErrors(cudnnSetTensor4dDescriptor(this->m_outputTd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               dims.batch, dims.channels, dims.height, dims.width));
}

template <typename T>
void InputLayer<T>::AllocateBuffers()
{
    checkCudaErrors(cudaMalloc(&this->m_outputBuffer, this->m_inputDims.Length()*sizeof(T)));
}

}

template class DLFS::InputLayer<float>;