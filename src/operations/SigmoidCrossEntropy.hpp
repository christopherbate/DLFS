#pragma once

#include "BaseOperation.hpp"
#include "GPU.hpp"
#include "tensor/Tensor.hpp"

namespace DLFS {

/**
 * Custom kernel launch functions.
 */
extern "C" void
LaunchSigmoidCEKernel(CustomOpDataType dataType, TensorShape logitShape,
                      const void *logits, TensorShape labelShape,
                      const void *labels, TensorShape outputShape,
                      const void *output, bool reduce_mean);

extern "C" void
LaunchSigmoidCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape,
                              void *logits, TensorShape labelShape,
                              void *labels, TensorShape outputShape,
                              void *output, bool reduce_mean);

/**
 * Implments SigmoidCrossEntropyLoss
 *
 * The inputs are assumed to be:
 * 1. Logits as floating points
 * 2. Integer indices for labels corresponding to indices
 *   in the logits vector
 *
 * The shape of the input A must be:
 * Batch x 1 x 1 x NumClasses
 *
 * The output, without reduction, is
 * Batch x 1 x 1 x NumClasses
 *
 * With class_reduction is:
 * Batch x 1 x 1 x 1
 *
 * And with batch reduction is:
 * 1 x 1 x 1 x 1
 */

template <typename T> class SigmoidCrossEntropyOp : public BaseOperation {
  public:
    SigmoidCrossEntropyOp() {}
    ~SigmoidCrossEntropyOp() {}

    void ExecuteForward() { Forward(); }

    void ExecuteBackward() { Backward(); }

    TensorBasePtr GetOutputTensor() { return m_output; }

    inline void SetLogits(TensorPtr<T> p) { m_inputA = p; }

    inline void SetLabels(TensorPtr<uint32_t> p) { m_labelTensor = p; }

    inline void SetOutput(TensorPtr<T> p) { m_output = p; }    

    /**
     * Enables mean-reduction immediately after calculating the loss.
     */
    void SetReduceMean(bool reduce_mean) {m_reduceMean = reduce_mean;}

  private:
    TensorPtr<T> m_inputA{nullptr};
    TensorPtr<uint32_t> m_labelTensor{nullptr};
    TensorPtr<T> m_output{nullptr};
    bool m_reduceMean{true};

  private:
    void Forward() {
        assert(m_inputA != nullptr);
        assert(m_labelTensor != nullptr);
        assert(m_output != nullptr);
        // m_output = CreateTensor

        LaunchSigmoidCEKernel(
            Float, m_inputA->GetShape(), m_inputA->GetDevicePointer(),
            m_labelTensor->GetShape(), m_labelTensor->GetDevicePointer(),
            m_output->GetShape(), m_output->GetDevicePointer(), m_reduceMean);

        cudaDeviceSynchronize();
    }

    void Backward() {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);
        assert(m_labelTensor != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        LaunchSigmoidCEBackwardKernel(
            Float, m_inputA->GetShape(), m_inputA->GetDevicePointer(),
            m_labelTensor->GetShape(), m_labelTensor->GetDevicePointer(),
            m_output->GetShape(), m_inputA->GetGradPointer(), m_reduceMean);
        cudaDeviceSynchronize();
        m_inputA->IncrementBackwardPass();
    }
};

template <typename T>
using SigmoidCEOpPtr = std::shared_ptr<SigmoidCrossEntropyOp<T>>;

} // namespace DLFS