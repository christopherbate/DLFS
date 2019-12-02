#pragma once

#include "BaseOperation.hpp"
#include "tensor/Tensor.hpp"
#include "GPU.hpp"

namespace DLFS
{

template <typename T>
class SoftmaxOp : public BaseOperation
{
public:
    SoftmaxOp() {}
    ~SoftmaxOp() {}

    void ExecuteForward()
    {
        Forward();
    }

    void ExecuteBackward()
    {
        Backward();
    }

    TensorBasePtr GetOutputTensor()
    {
        return m_output;
    }

    inline void SetInput(TensorPtr<T> p)
    {
        m_inputA = p;
    }

    inline void SetOutput(TensorPtr<T> p)
    {
        m_output = p;
    }

private:
    cudnnSoftmaxAlgorithm_t m_softmaxAlgorithm{CUDNN_SOFTMAX_ACCURATE};
    cudnnSoftmaxMode_t m_softmaxMode{CUDNN_SOFTMAX_MODE_CHANNEL};

    TensorPtr<T> m_inputA{nullptr};
    TensorPtr<T> m_output{nullptr};

private:
    void Forward()
    {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);

        const float blend[2] = {1.0, 0.0};

        LOG.DEBUG() << "Executing Softmax kernel.";

        checkCudaErrors(cudnnSoftmaxForward(
            GPUContext.GetCUDNNHandle(),
            m_softmaxAlgorithm,
            m_softmaxMode,
            (const void *)&blend[0],
            m_inputA->GetTensorDesc(),
            m_inputA->GetDevicePointer(),
            (const void *)&blend[1],
            m_output->GetTensorDesc(),
            m_output->GetDevicePointer()));

        cudaDeviceSynchronize();        
    }

    void Backward()
    {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);       
    }
};

template <typename T>
using SoftmaxOpPtr = std::shared_ptr<SoftmaxOp<T>>;

} // namespace DLFS