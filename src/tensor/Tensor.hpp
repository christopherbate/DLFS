#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <nvjpeg.h>
#include <vector>

#include "../Logging.hpp"

namespace DLFS
{

typedef std::array<int, 4> TensorShape;

/**
 * Tensor represents an array of 16-bit floating point numbers
 */
class Tensor
{
public:
    Tensor();
    ~Tensor();

    void SetShape(TensorShape shape);
    void SetShape(int b, int h, int w, int c);

    void AllocateIfNecessary();

    void Allocate();

    void Deallocate();

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

    inline const std::array<int, 4> &GetShape()
    {
        return m_shape;
    }

    inline size_t GetExpectedSize()
    {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3])*m_pitch;
    }

    inline size_t GetExpectedSizeWithinBatch(){
        return (m_shape[1] * m_shape[2] * m_shape[3])*m_pitch;
    }

private:
    TensorShape m_shape;
    size_t m_bufferSize;
    size_t m_pitch;
    uint8_t *m_deviceBuffer;
};

} // namespace DLFS

#endif