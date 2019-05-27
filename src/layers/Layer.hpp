#ifndef LAYER_H_
#define LAYER_H_

#include <cudnn.h>

#include "../Tensor.hpp"

namespace DLFS{
    
template <typename T>
class Layer
{
public:
    Layer(cudnnHandle_t &handle, Layer<T> *prevLayer);
    virtual ~Layer();

    virtual void Forward();
    virtual void Backward();

    cudnnTensorDescriptor_t GetOutputTensorDesc()
    {
        return m_outputTd;
    }

    const TensorDims &GetOutputDims()
    {
        return m_outputDims;
    }

    const TensorDims &GetInputDims()
    {
        return m_inputDims;
    }

    virtual void SetInputDim(TensorDims &inputDims){
        m_inputDims = inputDims;
    }

    virtual size_t GetMemoryRequirements()
    {
        return sizeof(T)*(m_inputDims.Length()+m_outputDims.Length());
    }    

    virtual void AllocateBuffers(){};   

    uint8_t *GetOutputBuffer(){
        return m_outputBuffer;
    } 

protected:
    cudnnHandle_t &m_handle;
    TensorDims m_inputDims;
    TensorDims m_outputDims;
    Layer<T> *m_prevLayer;
    cudnnTensorDescriptor_t m_outputTd;
    uint8_t *m_outputBuffer;    
};
}

#endif // !LAYER_H_