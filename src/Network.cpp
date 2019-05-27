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

template <typename T>
void Network<T>::Forward(NHWCBuffer<T> &inputData)
{
    std::cout << "Performing forward propagation ...\n";

    T *srcData = NULL;
    T *dstData = NULL;

    // // Copy the input data to the GPU
    // checkCudaErrors( cudaMalloc( &srcData, sizeof(T)*inputData.size() ) );
    // checkCudaErrors( cudaMemcpy( (void*)srcData, inputData.data(), inputData.size()*sizeof(T), cudaMemcpyHostToDevice) );

    // // Perform computations
    // NHWCBuffer<T> &nextInput = inputData;
    // for( int i = 0; i < m_layers.size(); i++)
    // {
    //     m_layers[i].Forward( nextInput );
    // }

    // // Copy output data back to RAM

    // // Free the gpu memory allocated for the input.
    // checkCudaErrors( cudaFree(srcData) );
    // checkCudaErrors( cudaFree(dstData) );
}

template <typename T>
void Network<T>::CreateTestTensor(NHWCBuffer<T> &output)
{
    const unsigned width = 4;
    const unsigned height = 4;
    const unsigned channels = 3;
    output.resize(m_batchSize * width * height * channels);
    const unsigned imgSize = width * height * channels;

    CBuffer<T> c1 = {1.0, 0.0, 1.0};
    CBuffer<T> c2 = {0.0, 3.0, 0.0};

    WCBuffer<T> row1 = {c1, c2, c1, c2};
    WCBuffer<T> row2 = {c2, c1, c2, c1};

    HWCBuffer<T> img1 = {row1, row2, row1, row2};

    for (int i = 0; i < m_batchSize; i++)
    {
        output.push_back(img1);
    }
}
}

template class DLFS::Network<float>;