#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <nvjpeg.h>

namespace DLFS 
{
/**
 * Tensor represents an array of 16-bit floating point numbers
 */
class Tensor {
public:
    Tensor():
    m_bufferSize(0), m_deviceBuffer(NULL) {
    }

    void SetShape(std::array<int, 4> &shape){
        m_shape = shape;
    }

    void SetShape(int b, int h, int w, int c){
        m_shape = std::array<int,4>{b,h,w,c};
    }

    void AllocateIfNecessary(){

    }

    unsigned char* GetPointer(int batch, int channel)
    {

    }


private:
    std::array<int, 4> m_shape{1,1,1,1};
    size_t m_bufferSize;
    uint8_t *m_deviceBuffer;
};
}

#endif