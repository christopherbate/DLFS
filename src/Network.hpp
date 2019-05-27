#ifndef NETWORK_H_
#define NETWORK_H_

#include <cudnn.h>

#include <iostream>
#include <string>
#include <vector>

#include "./layers/Layer.hpp"

namespace DLFS
{
template <typename T>
class Network
{
private:
    cudnnHandle_t m_cudnnHandle;
    std::vector<Layer<T>> m_layers;
    unsigned int m_batchSize;

public:
    Network();
    ~Network();

    cudnnHandle_t &GetCUDNN()
    {
        return m_cudnnHandle;
    }

    void Forward(NHWCBuffer<T> &inputData);
    void Backward();

    void CreateTestTensor(NHWCBuffer<T> &output);
};

} // namespace DLFS

#endif // !NETWORK_H_
