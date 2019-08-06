#include "Tensor.hpp"
#include <iostream>
#include <vector>

using namespace DLFS;
using namespace std;

Tensor::Tensor()
    : m_bufferSize(0), m_deviceBuffer(NULL), m_pitch(1)
{
    m_shape = {1, 1, 1, 1};
    m_bufferSize = 1;
}

Tensor::~Tensor()
{
    Deallocate();
}

void Tensor::SetShape(TensorShape shape)
{
    m_shape = shape;
}

void Tensor::SetShape(int b, int h, int w, int c)
{
    m_shape = std::array<int, 4>{b, h, w, c};
}

void Tensor::AllocateIfNecessary()
{
    size_t needed_bytes = GetExpectedSize();
    if (needed_bytes > m_bufferSize || m_deviceBuffer == NULL)
    {
        cout << "Need to reallocate buffer: " << needed_bytes << " curr: " << m_bufferSize << " ptr: " << uint64_t(m_deviceBuffer) << endl;
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
        cout << "De allocating buffer: " << uint64_t(m_deviceBuffer) << endl;
        checkCudaErrors(cudaFree(m_deviceBuffer));
        m_deviceBuffer = NULL;
    }
}