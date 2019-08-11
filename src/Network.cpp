#include "Network.hpp"
#include "Logging.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <string>
#include <sstream>


using namespace std;


namespace DLFS
{

/**
 * Neural Network
 **/

template <typename T>
Network<T>::Network()
{
    checkCudaErrors(cudnnCreate(&m_cudnnHandle));
    m_batchSize = 1;
}

template <typename T>
Network<T>::~Network()
{
    checkCudaErrors(cudnnDestroy(m_cudnnHandle));
}
}

template class DLFS::Network<float>;