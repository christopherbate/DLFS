#pragma once

#include "tensor/AutoDiff.hpp"
#include "tensor/Tensor.hpp"
#include "GPU.hpp"

namespace DLFS
{

/**
 * Implments pointwise Op( scale[0]*A, scale[1]*B) + scale[2]*C
 * 
*/
enum PointwiseOpType
{
    PW_ADD = CUDNN_OP_TENSOR_ADD,
    PW_MUL = CUDNN_OP_TENSOR_MUL,
    PW_MAX = CUDNN_OP_TENSOR_MAX,
    PW_MIN = CUDNN_OP_TENSOR_MIN,
    PW_SQRT = CUDNN_OP_TENSOR_SQRT
};

template <typename T>
class TensorOp : public TrackableOp
{
public:
    TensorOp(PointwiseOpType t)
    {
        m_tensorOp = static_cast<cudnnOpTensorOp_t>(t);
        checkCudaErrors(cudnnCreateOpTensorDescriptor(&m_tensorOpDesc));
    }
    ~TensorOp()
    {
        if (m_tensorOpDesc != nullptr)
            checkCudaErrors(cudnnDestroyOpTensorDescriptor(m_tensorOpDesc));
    }    

    void ExecuteForward()
    {
        assert(m_inputA != nullptr);
        assert(m_inputB != nullptr);
        assert(m_output != nullptr);

        checkCudaErrors(cudnnSetOpTensorDescriptor(m_tensorOpDesc,
                                                   CUDNN_OP_TENSOR_ADD,
                                                   CUDNN_DATA_FLOAT,
                                                   CUDNN_PROPAGATE_NAN));

        checkCudaErrors(cudnnOpTensor(
            GPUContext.GetCUDNNHandle(),
            m_tensorOpDesc,
            &m_scaleFactors[0],
            m_inputA->GetTensorDesc(),
            m_inputA->GetPointer(),
            &m_scaleFactors[1],
            m_inputB->GetTensorDesc(),
            m_inputB->GetPointer(),
            &m_scaleFactors[2],
            m_output->GetTensorDesc(),
            m_output->GetPointer()));
    }

    void ExecuteBackward()
    {
    }

    TensorBasePtr GetOutputTensor()
    {
        return m_output;
    }

    inline void SetScales(T a1, T a2, T b)
    {
        m_scaleFactors = {a1,a2,b};
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

private:
    cudnnOpTensorOp_t m_tensorOp;
    cudnnOpTensorDescriptor_t m_tensorOpDesc;
    std::array<T, 3> m_scaleFactors{{1.0, 1.0, 1.0}};
    TensorPtr<T> m_inputA{nullptr};
    TensorPtr<T> m_inputB{nullptr};
    TensorPtr<T> m_output{nullptr};
};

template <typename T>
using TensorOpPtr = std::shared_ptr<TensorOp<T>>;

} // namespace DLFS