#include "Layer.hpp"
#include "../Logging.hpp"

namespace DLFS
{

template <typename T>
Layer<T>::Layer(cudnnHandle_t &handle) : m_handle(handle)
{
    m_outputTd = NULL;
    m_outputBuffer = NULL;

    checkCudaErrors(cudnnCreateTensorDescriptor(&m_outputTd));
}

template <typename T>
Layer<T>::~Layer()
{
    if (m_outputTd)
    {
        checkCudaErrors(cudnnDestroyTensorDescriptor(m_outputTd));
        m_outputTd = NULL;
    }

    if (m_outputBuffer)
    {
        checkCudaErrors(cudaFree(m_outputBuffer));
        m_outputBuffer = NULL;
    }
}

template <typename T>
void Layer<T>::SetInputLayer(Layer<T> *inputLayer)
{
    // m_prevLayer = inputLayer;
    // if (inputLayer)
    // {        
        // m_inputDims = m_prevLayer->GetOutputDims();
    // }            
}

template <typename T>
void Layer<T>::Forward()
{
}
template <typename T>
void Layer<T>::Backward()
{
}

} // namespace DLFS

template class DLFS::Layer<float>;