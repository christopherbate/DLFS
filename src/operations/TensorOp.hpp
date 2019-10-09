#pragma once

#include "BaseOperation.hpp"
#include "tensor/Tensor.hpp"
#include "GPU.hpp"

namespace DLFS
{

/**
* Custom kernel launch functions.
*/
extern "C" void LaunchPowerKernel(CustomOpDataType dataType,
                       void *inputBuffer, int linearLength,
                       void *power, void *outputBuffer);

/**
 * Implments pointwise Op( scale[0]*A, scale[1]*B) + scale[2]*C
 * 
*/
enum PointwiseOpType : uint32_t
{
    PW_ADD = CUDNN_OP_TENSOR_ADD,
    PW_MUL = CUDNN_OP_TENSOR_MUL,
    PW_MAX = CUDNN_OP_TENSOR_MAX,
    PW_MIN = CUDNN_OP_TENSOR_MIN,
    PW_SQRT = CUDNN_OP_TENSOR_SQRT,
    PW_POW    
};

template <typename T>
class TensorOp : public BaseOperation
{
public:
    TensorOp(PointwiseOpType opType)
    {
        m_pointwiseOpType = opType;
        checkCudaErrors(cudnnCreateOpTensorDescriptor(&m_tensorOpDesc));       
    }
    ~TensorOp()
    {
        if (m_tensorOpDesc != nullptr)
            checkCudaErrors(cudnnDestroyOpTensorDescriptor(m_tensorOpDesc));
    }

    void ExecuteForward()
    {       
        switch (m_pointwiseOpType)
        {
        case PW_ADD:
            ForwardAdd();
            break;

        case PW_POW:
            ForwardPow();
            break;

        default:
            throw std::runtime_error("Pointwise forward op not implemented.");
            break;
        }
    }

    void ExecuteBackward()
    {        
        switch(m_pointwiseOpType)
        {
            case PW_ADD:
                BackwardAdd();
                break;

            case PW_POW:
                BackwardPow();
                break;

            default:
                throw std::runtime_error("Pointwise backward op not implemented.");
                break;
        }      
    }

    TensorBasePtr GetOutputTensor()
    {
        return m_output;
    }

    inline void SetScales(T a1, T a2, T b)
    {
        m_scaleFactors = {a1, a2, b};
    }

    inline void SetInput(TensorPtr<T> p, unsigned int idx)
    {
        if (idx == 0)
            m_inputA = p;
        else if (idx == 1)
            m_inputB = p;
    }

    inline void SetOutput(TensorPtr<T> p)
    {
        m_output = p;
    }

    inline void SetPower(T power)
    {
        m_power = power;
    }

private:
    PointwiseOpType m_pointwiseOpType;
    cudnnOpTensorOp_t m_tensorOp;
    cudnnOpTensorDescriptor_t m_tensorOpDesc;
    std::array<T, 3> m_scaleFactors{{1, 1, 1}};
    TensorPtr<T> m_inputA{nullptr};
    TensorPtr<T> m_inputB{nullptr};
    TensorPtr<T> m_output{nullptr};
    T m_power{1};

private:
    void ForwardAdd()
    {
        assert(m_inputA != nullptr);
        assert(m_inputB != nullptr);
        assert(m_output != nullptr);

        // for(auto i=(unsigned)0;i<4; i++)
        // {
        //     assert(m_inputA->GetShape()[i] == m_inputB->GetShape()[i]);
        // }

        checkCudaErrors(cudnnSetOpTensorDescriptor(m_tensorOpDesc,
                                                   CUDNN_OP_TENSOR_ADD,
                                                   CUDNN_DATA_FLOAT,
                                                   CUDNN_PROPAGATE_NAN));
       
        checkCudaErrors(cudnnOpTensor(
            GPUContext.GetCUDNNHandle(),
            m_tensorOpDesc,
            &m_scaleFactors[0],
            m_inputA->GetTensorDesc(),
            m_inputA->GetDevicePointer(),
            &m_scaleFactors[1],
            m_inputB->GetTensorDesc(),
            m_inputB->GetDevicePointer(),
            &m_scaleFactors[2],
            m_output->GetTensorDesc(),
            m_output->GetDevicePointer()));

        cudaDeviceSynchronize();
    }

    void BackwardAdd()
    {
        assert(m_inputA != nullptr);
        assert(m_inputB != nullptr);
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        std::vector<TensorPtr<T>> inputs = {m_inputA, m_inputB};
        for (auto inputT : inputs)
        {
                if (!inputT->GetGradFlag())
                    continue;

                assert(inputT->GetGradPointer() != nullptr);
                // a few simplifications we can make.
                // if this is an "add" op, just append
                // the output's grad
                T blendFactors[3] = {1, 0, 1};
                if (inputT->IsFirstBackwardPass())
                {
                    blendFactors[2] = 0;
                }
                inputT->IncrementBackwardPass();
                
                checkCudaErrors(cudnnSetOpTensorDescriptor(m_tensorOpDesc,
                                                   CUDNN_OP_TENSOR_ADD,
                                                   CUDNN_DATA_FLOAT,
                                                   CUDNN_PROPAGATE_NAN));

                checkCudaErrors(cudnnOpTensor(
                    GPUContext.GetCUDNNHandle(),
                    m_tensorOpDesc,
                    &blendFactors[0],
                    m_output->GetTensorDesc(),
                    m_output->GetGradPointer(),
                    &blendFactors[1],
                    m_output->GetTensorDesc(),
                    m_output->GetGradPointer(),
                    &blendFactors[2],
                    inputT->GetTensorDesc(),
                    inputT->GetGradPointer()));
        }
    }

    void ForwardPow()
    {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);

        LaunchPowerKernel(CustomOpDataType::Float,
                          m_inputA->GetDevicePointer(),
                          m_inputA->GetLinearSize(),
                          &m_power, m_output->GetDevicePointer());                          
        cudaDeviceSynchronize();
    }

    void BackwardPow()
    {
        assert(m_inputA != nullptr);        
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);
        assert(m_inputA->GetGradPointer() != nullptr);

        T blendFactors[3] = {m_power, 1, 1};
        if (m_inputA->IsFirstBackwardPass())
        {
            blendFactors[2] = 0;
        }
        m_inputA->IncrementBackwardPass();


        // Gradients is y*(x)^(y-1)*(prevDelta)
        // We take care of the exponent first:
        T dPower = m_power - 1;
        std::unique_ptr<Tensor<T>> tmp = std::make_unique<Tensor<T>>();
        tmp->SetShape(m_inputA->GetShape());
        tmp->AllocateIfNecessary();

        LaunchPowerKernel(CustomOpDataType::Float,
                          m_inputA->GetDevicePointer(),
                          m_inputA->GetLinearSize(),
                          &dPower,
                          tmp->GetDevicePointer());

        checkCudaErrors(cudnnSetOpTensorDescriptor(m_tensorOpDesc,
                                                   CUDNN_OP_TENSOR_MUL,
                                                   CUDNN_DATA_FLOAT,
                                                   CUDNN_PROPAGATE_NAN));

        // Next take care of the multiplications        
        checkCudaErrors(cudnnOpTensor(
                    GPUContext.GetCUDNNHandle(),
                    m_tensorOpDesc,
                    &blendFactors[0],
                    tmp->GetTensorDesc(),
                    tmp->GetDevicePointer(),
                    &blendFactors[1],
                    m_output->GetTensorDesc(),
                    m_output->GetGradPointer(),
                    &blendFactors[2],
                    m_inputA->GetTensorDesc(),
                    m_inputA->GetGradPointer()));
    }
};

template <typename T>
using TensorOpPtr = std::shared_ptr<TensorOp<T>>;

} // namespace DLFS