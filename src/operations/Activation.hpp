#pragma once

#include "BaseOperation.hpp"
#include "GPU.hpp"
#include "tensor/Tensor.hpp"

namespace DLFS {

/**
 * Custom kernel launch functions.
 */

/**
 * Implments Activation Function
 *
 * Only ReLU is implemented right now.
 *
 */

template <typename T> class ActivationOp : public BaseOperation {
  public:
    ActivationOp() {
        checkCudaErrors(cudnnCreateActivationDescriptor(&m_activationDesc));
        checkCudaErrors(cudnnSetActivationDescriptor(
            m_activationDesc, m_activationMode, CUDNN_PROPAGATE_NAN, 10.0));
    }
    ~ActivationOp() {
        checkCudaErrors(cudnnDestroyActivationDescriptor(m_activationDesc));
    }

    void ExecuteForward() {
        assert(m_input != nullptr);
        assert(m_output != nullptr);

        T blend[2] = {1, 0};
        checkCudaErrors(cudnnActivationForward(
            GPUContext.GetCUDNNHandle(), m_activationDesc, &blend[0],
            m_input->GetTensorDesc(), m_input->GetDevicePointer(), &blend[1],
            m_output->GetTensorDesc(), m_output->GetDevicePointer()));

        cudaDeviceSynchronize();
    }

    void ExecuteBackward() {
        assert(m_input != nullptr);
        assert(m_output != nullptr);
        assert(m_output->GetGradPointer() != nullptr);

        T blend[2] = {1, 0};

        checkCudaErrors(cudnnActivationBackward(
            GPUContext.GetCUDNNHandle(), m_activationDesc, &blend[0],
            m_output->GetTensorDesc(), m_output->GetDevicePointer(),
            m_output->GetTensorDesc(), m_output->GetGradPointer(),
            m_input->GetTensorDesc(), m_output->GetDevicePointer(),
            &blend[1],
            m_input->GetTensorDesc(), m_input->GetGradPointer()));

        cudaDeviceSynchronize();

        m_input->IncrementBackwardPass();
    }

    TensorBasePtr GetOutputTensor() { return m_output; }

    inline void SetInput(TensorPtr<T> p) { m_input = p; }

    inline void SetOutput(TensorPtr<T> p) { m_output = p; }

  private:
    TensorPtr<T> m_input{nullptr};
    TensorPtr<T> m_output{nullptr};
    bool m_inplace{false};

    cudnnActivationDescriptor_t m_activationDesc;
    cudnnActivationMode_t m_activationMode{CUDNN_ACTIVATION_RELU};
};

template <typename T> using ActivationOpPtr = std::shared_ptr<ActivationOp<T>>;

} // namespace DLFS