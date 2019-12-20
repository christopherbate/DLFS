#pragma once

#include "BaseOperation.hpp"
#include "GPU.hpp"
#include "tensor/Tensor.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

namespace DLFS {
/**
 * Implments BatchNorm Function
 *
 * This is equivalent to "BatchNorm2D" in PyTorch, which
 * means this covers "spatial" batch normalization. The entire WxH for each
 * channel are assumed to be pointwise from the same distribution.
 */
template <typename DataType> class BatchNormOp : public BaseOperation {
  public:
    BatchNormOp(TensorPtr<DataType> input) {
        checkCudaErrors(cudnnCreateTensorDescriptor(&m_paramDesc));
        checkCudaErrors(cudnnDeriveBNTensorDescriptor(
            m_paramDesc, input->GetTensorDesc(), m_batchNormMode));

        m_input = input;

        auto paramShape = TensorShape{1, 1, 1, input->GetShape()[3]};
        m_scale = CreateTensor(paramShape, "bn-scale", DataType(1), true);
        m_bias = CreateTensor(paramShape, "bn-scale", DataType(0), true);

        m_output =
            CreateTensor(input->GetShape(), "bn-output", DataType(0), true);
    }
    ~BatchNormOp() {
        checkCudaErrors(cudnnDestroyTensorDescriptor(m_paramDesc));
    }

    TensorPtr<DataType> ExecuteForward() {
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        DataType blendFactors[2] = {DataType(1), DataType(0)};
        checkCudaErrors(cudnnBatchNormalizationForwardTraining(
            GPUContext.GetCUDNNHandle(), m_batchNormMode, &blendFactors[0],
            &blendFactors[1], m_input->GetTensorDesc(),
            m_input->GetDevicePointer(), m_output->GetTensorDesc(),
            m_output->GetDevicePointer(), m_paramDesc,
            m_scale->GetDevicePointer(), m_bias->GetDevicePointer(), m_momentum,
            /*resultsRunningMean (ret value if needed)*/ nullptr,
            /*resultsRunningVariance (ret value if needed)*/ nullptr,
            /* variance additive epsilon */ m_epsilon,
            /*resultSaveMean (cudnn cache  value)*/ nullptr,
            /*resultSaveInvVariance (cudnn cached value)*/ nullptr));
        return m_output;
    }

    void ExecuteBackward() {
        assert(m_input != nullptr);
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        DataType blend[2] = {1, 0};

        checkCudaErrors(cudnnBatchNormalizationBackward(
            GPUContext.GetCUDNNHandle(), m_batchNormMode, &blend[0], &blend[1],
            &blend[0], &blend[1], m_input->GetTensorDesc(),
            m_input->GetDevicePointer(), m_output->GetTensorDesc(),
            m_output->GetGradPointer(), m_input->GetTensorDesc(),
            m_input->GetGradPointer(), m_paramDesc, m_scale->GetDevicePointer(),
            m_scale->GetGradPointer(), m_bias->GetGradPointer(), m_epsilon,
            nullptr, nullptr));

        checkCudaErrors(cudaDeviceSynchronize());
        m_input->IncrementBackwardPass();
    }

    TensorBasePtr GetOutputTensor() { return m_output; }

    TensorPtr<DataType> GetScaleTensor() { return m_scale; }
    TensorPtr<DataType> GetBiasTensor() { return m_bias; }

  private:
    TensorPtr<DataType> m_input{nullptr};
    TensorPtr<DataType> m_output{nullptr};

    /** Scale and bias parameters (elementwise) **/
    TensorPtr<DataType> m_scale{nullptr};
    TensorPtr<DataType> m_bias{nullptr};

    /** Meand and vairnace (intermediate computations) **/
    TensorPtr<DataType> m_mean{nullptr};
    TensorPtr<DataType> m_variance{nullptr};

    /** Epsilon value - added to the variance before square root **/
    double m_epsilon{1e-5};

    /** Exponential moving average factor **/
    double m_momentum{0.1f};

    cudnnTensorDescriptor_t m_paramDesc;
    cudnnBatchNormMode_t m_batchNormMode{CUDNN_BATCHNORM_SPATIAL_PERSISTENT};
};

template <typename DataType>
using BatchNormOpPtr = std::shared_ptr<BatchNormOp<DataType>>;

template <typename DataType>
TensorPtr<DataType> MakeBatchNorm(TensorPtr<DataType> input) {
    auto op = std::make_shared<BatchNormOp<DataType>>(input);
    auto output = op->ExecuteForward();
    return output;
}

} // namespace DLFS