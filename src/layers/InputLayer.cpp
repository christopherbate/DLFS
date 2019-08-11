#include "InputLayer.hpp"
#include "../Logging.hpp"

namespace DLFS
{

/**
 * InputLayer
 **/
template <typename T>
InputLayer<T>::InputLayer(cudnnHandle_t &handle) : Layer<T>(handle)
{
}

template <typename T>
InputLayer<T>::~InputLayer()
{
}
   

template <typename T>
void InputLayer<T>::AllocateOutputBuffers()
{
  
}

template class DLFS::InputLayer<float>;
}