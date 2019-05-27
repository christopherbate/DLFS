#include "Layer.hpp"
#include "../Logging.hpp"

namespace DLFS{

template <typename T>
Layer<T>::Layer(cudnnHandle_t &handle, Layer<T> *prevLayer) : m_handle(handle), m_prevLayer(prevLayer)
{
    if(prevLayer){
        m_inputDims = m_prevLayer->GetOutputDims();        
    }
    checkCudaErrors(cudnnCreateTensorDescriptor(&m_outputTd));
    m_outputBuffer = NULL;
}

template <typename T>
Layer<T>::~Layer()
{
    checkCudaErrors(cudnnDestroyTensorDescriptor(m_outputTd));
    if(m_outputBuffer){
        checkCudaErrors(cudaFree(m_outputBuffer));
        m_outputBuffer = NULL;
    }
}

template <typename T>
void Layer<T>::Forward()
{
    
}
template <typename T>
void Layer<T>::Backward()
{    
}

}

template class DLFS::Layer<float>;