#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "../Logging.hpp"

namespace DLFS
{

typedef std::array<int, 4> TensorShape;

/**
 * Tensor represents an array of 16-bit floating point numbers
 */
class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    Tensor();
    ~Tensor();

    void SetShape(TensorShape shape);
    void SetShape(int b, int h, int w, int c);
    void SetFilterShape(int numInputChn,
                        int numOutputChn,
                        int rows, int cols);
    void FillCUDNNDesc();

    void AllocateIfNecessary();

  

    std::shared_ptr<Tensor> Convolve(std::shared_ptr<Tensor> filter);

    inline void SetPitch(size_t pitch)
    {
        m_pitch = pitch;
    }

    inline std::vector<unsigned char *> GetIterablePointersOverBatch()
    {
        std::vector<unsigned char *> batchPointers(m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3];
        for (int i = 0; i < m_shape[0]; i++)
        {
            batchPointers[i] = m_deviceBuffer + itemSize * i;
        }
        return batchPointers;
    }

    inline unsigned char *GetPointer()
    {
        return (unsigned char *)m_deviceBuffer;
    }

    inline std::array<int, 4> GetShape() const
    {
        return m_shape;
    }

    inline size_t GetExpectedSize()
    {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline size_t GetExpectedSizeWithinBatch()
    {
        return (m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline cudnnTensorDescriptor_t GetTensorDesc()
    {
        return m_tensorDesc;
    }

    inline cudnnFilterDescriptor_t GetFilterDesc()
    {
        return m_filterDesc;
    }

    inline std::string PrintShape()
    {
        std::string shapeMsg = "(";
        for (auto d : m_shape)
            shapeMsg += std::to_string(d) + ", ";
        shapeMsg += ")";
        return shapeMsg;
    }

private:
    void Allocate();
    void Deallocate();

private:
    TensorShape m_shape;
    cudnnDataType_t m_dataType;
    cudnnTensorDescriptor_t m_tensorDesc;
    cudnnFilterDescriptor_t m_filterDesc;
    size_t m_bufferSize;
    uint8_t *m_deviceBuffer;
    size_t m_pitch;
    bool m_isFilter;
};

typedef std::shared_ptr<Tensor> TensorPtr;

} // namespace DLFS
