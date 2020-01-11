#pragma once

#include "BaseOperation.hpp"
#include "../GPU.hpp"
#include "../tensor/Tensor.hpp"

namespace DLFS {

extern "C" void LaunchSoftmaxCEKernel(CustomOpDataType dataType,
                                      TensorShape logitShape, void *logits,
                                      TensorShape labelShape, void *labels,
                                      TensorShape outputShape, void *output,
                                      bool reduce_mean);

extern "C" void
LaunchSoftmaxCEBackwardKernel(CustomOpDataType dataType, TensorShape logitShape,
                              void *logits, TensorShape labelShape,
                              void *labels, TensorShape outputShape,
                              void *output);

template <typename T> class SoftmaxOp : public BaseOperation {
  public:
    SoftmaxOp() {}
    ~SoftmaxOp() {}

    void ExecuteForward() { Forward(); }

    void ExecuteBackward() { Backward(); }

    TensorBasePtr GetOutputTensor() { return m_output; }

    inline void SetInput(TensorPtr<T> p) { m_inputA = p; }

    inline void SetOutput(TensorPtr<T> p) { m_output = p; }

  private:
    cudnnSoftmaxAlgorithm_t m_softmaxAlgorithm{CUDNN_SOFTMAX_ACCURATE};
    cudnnSoftmaxMode_t m_softmaxMode{CUDNN_SOFTMAX_MODE_CHANNEL};

    TensorPtr<T> m_inputA{nullptr};
    TensorPtr<T> m_output{nullptr};

  private:
    void Forward() {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);

        const T blend[2] = {1, 0};

        LOG.DEBUG() << "Executing Softmax kernel.";

        checkCudaErrors(cudnnSoftmaxForward(
            GPUContext.GetCUDNNHandle(), m_softmaxAlgorithm, m_softmaxMode,
            &blend[0], m_inputA->GetTensorDesc(), m_inputA->GetDevicePointer(),
            &blend[1], m_output->GetTensorDesc(),
            m_output->GetDevicePointer()));

        checkCudaErrors(cudaDeviceSynchronize());
    }

    void Backward() {
        assert(m_inputA != nullptr);
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);
    }
};

template <typename T> using SoftmaxOpPtr = std::shared_ptr<SoftmaxOp<T>>;

template <typename T> class SoftmaxCELossOp : public BaseOperation {
  public:
    SoftmaxCELossOp() {}
    ~SoftmaxCELossOp() {}

    void ExecuteForward() { Forward(); }

    void ExecuteBackward() { Backward(); }

    TensorBasePtr GetOutputTensor() { return m_output; }

    inline void SetLogits(TensorPtr<T> p) { m_inputA = p; }

    inline void SetLabels(TensorPtr<uint32_t> p) { m_labelTensor = p; }

    inline void SetOutput(TensorPtr<T> p) { m_output = p; }

    /**
     * Enables mean-reduction immediately after calculating the loss.
     */
    void SetReduce(bool reduce_mean) { m_reduceMean = reduce_mean; }

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

        LaunchSoftmaxCEKernel(
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

        LaunchSoftmaxCEBackwardKernel(
            Float, m_inputA->GetShape(), m_inputA->GetDevicePointer(),
            m_labelTensor->GetShape(), m_labelTensor->GetDevicePointer(),
            m_output->GetShape(), m_inputA->GetGradPointer());
        cudaDeviceSynchronize();
        m_inputA->IncrementBackwardPass();
    }
};

template <typename DataType>
using SoftmaxCELossOpPtr = std::shared_ptr<SoftmaxCELossOp<DataType>>;

template <typename DataType>
TensorPtr<DataType> SoftmaxCELoss(TensorPtr<DataType> logits,
                                  TensorPtr<uint32_t> labels,
                                  bool reduce = true) {
    SoftmaxCELossOpPtr<DataType> op =
        std::make_shared<SoftmaxCELossOp<DataType>>();
    op->SetName("SigmoidCEOp");
    op->SetLogits(logits);
    op->SetLabels(labels);

    auto outputShape = TensorShape{reduce ? 1 : logits->GetShape()[0], 1, 1, 1};

    TensorPtr<DataType> outputTensor =
        CreateTensor(outputShape, "softmaxCELossOutput", DataType(0), false);

    op->SetOutput(outputTensor);
    op->SetReduce(reduce);
    op->ExecuteForward();

    ADContext.AddOp(op);

    return outputTensor;
}

} // namespace DLFS