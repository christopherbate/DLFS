#ifndef LAYER_H_
#define LAYER_H_

#include <cudnn.h>

#include "../tensor/Tensor.hpp"

namespace DLFS{
    
template <typename T>
class Layer
{
public:
    Layer(cudnnHandle_t &handle);
    virtual ~Layer();

    virtual void Forward();
    virtual void Backward();

    virtual void SetInputLayer(Layer<T> *layer);    

    cudnnTensorDescriptor_t GetOutputTensorDesc()
    {
        return m_outputTd;
    }  

    virtual void AllocateOutputBuffers() = 0;
    virtual void AllocateWeightBuffers() = 0;

    uint8_t *GetOutputBuffer(){
        return m_outputBuffer;
    } 

protected:
    cudnnHandle_t &m_handle;
    Layer<T> *m_prevLayer;
    cudnnTensorDescriptor_t m_outputTd;
    uint8_t *m_outputBuffer;    
};
}

#endif // !LAYER_H_