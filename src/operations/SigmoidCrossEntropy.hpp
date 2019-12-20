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

extern "C" void LaunchSigmoidCEBackwardKernel(CustomOpDataType dataType,
                                              TensorShape logitShape,
                                              void *logits, void *dLogits,
                                              TensorShape labelShape,
                                              void *labels);

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
    SigmoidCrossEntropyOp(TensorPtr<T> logits, TensorPtr<uint32_t> labels,
                          bool reduce) {
        m_logits = logits;
        m_labels = labels;
        m_reduceMean = reduce;

        auto outputShape =
            reduce ? TensorShape{1, 1, 1, 1} : labels->GetShape();
        m_output =
            CreateTensor(outputShape, "sigmoid-ce-loss-output", T(0), true);
    }
    ~SigmoidCrossEntropyOp() {}

    void ExecuteForward() { Forward(); }

    void ExecuteBackward() { Backward(); }

    TensorBasePtr GetOutputTensor() { return m_output; }

    inline void SetLogits(TensorPtr<T> p) { m_logits = p; }

    inline void SetLabels(TensorPtr<uint32_t> p) { m_labels = p; }

    inline void SetOutput(TensorPtr<T> p) { m_output = p; }

    /**
     * Enables mean-reduction immediately after calculating the loss.
     */
    void SetReduceMean(bool reduce_mean) { m_reduceMean = reduce_mean; }

  private:
    TensorPtr<T> m_logits{nullptr};
    TensorPtr<uint32_t> m_labels{nullptr};
    TensorPtr<T> m_output{nullptr};
    bool m_reduceMean{true};

  public:
    TensorPtr<T> Forward() {
        assert(m_logits != nullptr);
        assert(m_labels != nullptr);
        assert(m_output != nullptr);
        // m_output = CreateTensor

        LaunchSigmoidCEKernel(
            Float, m_logits->GetShape(), m_logits->GetDevicePointer(),
            m_labels->GetShape(), m_labels->GetDevicePointer(),
            m_output->GetShape(), m_output->GetDevicePointer(), m_reduceMean);

        cudaDeviceSynchronize();

        return m_output;
    }

    void Backward() {
        assert(m_logits != nullptr);
        assert(m_output != nullptr);
        assert(m_labels != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        LaunchSigmoidCEBackwardKernel(
            Float, m_logits->GetShape(), m_logits->GetDevicePointer(),
            m_logits->GetGradPointer(), m_labels->GetShape(),
            m_labels->GetDevicePointer());
        cudaDeviceSynchronize();
        m_logits->IncrementBackwardPass();
    }
};

template <typename T>
using SigmoidCEOpPtr = std::shared_ptr<SigmoidCrossEntropyOp<T>>;

template <typename DataType>
TensorPtr<DataType> SigmoidCELoss(TensorPtr<DataType> logits,
                                  TensorPtr<uint32_t> labels,
                                  bool reduce = true) {
    SigmoidCEOpPtr<DataType> op =
        std::make_shared<SigmoidCrossEntropyOp<DataType>>(logits, labels,
                                                          reduce);
    op->SetName("SigmoidCELossOp");

    auto out = op->Forward();
    ADContext.AddOp(op);

    return out;
}

} // namespace DLFS