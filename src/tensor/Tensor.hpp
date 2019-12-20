/**
 * Tensor object implementation
 *
 * TensorBase implements required functions irrespective
 * of template/datatype (mostly things required by the autodiff
 * system)
 *
 * Tensor inherits from TensorBase and is a templated class
 * Tensor handles all internal cudnn buffers and has methods for
 * initialization, copying, and overloads certain arithmetic operators.
 *
 * The Tensor class's arithmetic methods automaticallly interface with
 * the global AutoDiff context in order to add the op and the input/output
 * tensors to the op stack.
 *
 * Christopher Bate
 * August/September 2019
 **/
#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "../Logging.hpp"
#include "GPU.hpp"
#include "TensorBase.hpp"
#include "operations/BaseOperation.hpp"
#include "operations/types.hpp"

namespace DLFS {
/*
 Fwd Declare so we can use the Tensor Ptr type in the
class decl.
*/
template <typename T> class Tensor;
template <typename T> using TensorPtr = std::shared_ptr<Tensor<T>>;

template <typename FeatureDataType, typename FilterDataType>
TensorPtr<FeatureDataType> MakeConvolve(TensorPtr<FeatureDataType> features,
                                        TensorPtr<FilterDataType> filter,
                                        Stride2D stride = {1, 1},
                                        Pad2D padding = {0, 0});

/**
 * Tensor represents an array of double/float/half/int numbers.
 */
template <typename T>
class Tensor : public TensorBase,
               public std::enable_shared_from_this<Tensor<T>> {
  public:
    Tensor(const std::string &name = "");
    ~Tensor();

    size_t GetDataTypeSize() override { return sizeof(T); }

    /**
     * Type-Casting
     * Returns a tensor with a new type.
     * TODO: CUDA kernel
     */
    template <typename TargetType> TensorPtr<TargetType> Cast();

    /**
     * Initialization Functions
     */

    /**
     *  Fills device-side buffer with constant value
     **/
    void FillConstant(T constVal) {
        assert(m_deviceBuffer != nullptr);

        std::vector<T> localBuffer(GetLinearSize(), constVal);
        checkCudaErrors(cudaMemcpy(m_deviceBuffer, localBuffer.data(),
                                   localBuffer.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    /**
     *  Fills gradient buffer with constant value
     **/
    void FillConstantGrad(T constVal) {
        assert(m_deviceBufferGrad != nullptr);
        std::vector<T> localBuffer(GetLinearSize(), constVal);
        checkCudaErrors(cudaMemcpy(m_deviceBufferGrad, localBuffer.data(),
                                   localBuffer.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    const std::string &PrintTensor(bool grad = false,
                                   bool breakChannels = true);

    void InitGradChain() {
        assert(m_deviceBuffer != nullptr);
        FillConstantGrad(1.0f);
        ResetBackwardPasses();
        IncrementBackwardPass();
    }

    /**
     * Buffer from must be the correct size.
     * device buffer must already have been allocated.
     */
    void CopyBufferToDevice(const std::vector<T> &from) {
        assert(from.size() == GetLinearSize());
        assert(m_deviceBuffer != nullptr);
        checkCudaErrors(cudaMemcpy(m_deviceBuffer, from.data(), GetSizeBytes(),
                                   cudaMemcpyHostToDevice));
    }

    /**
     * Accepts a vector of (host) buffers and copies them to the device buffer
     * along the batch dimension.
     *
     * The individual buffers from the host all need to be the size of the
     * sub-batch slice on the device.
     */
    void CopyBatchBuffersToDevice(const std::vector<std::vector<T>> &from) {
        assert((int)from.size() == m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3] * sizeof(T);
        for (int i = 0; i < m_shape[0]; i++) {
            assert((int)(from[i].size() * sizeof(T)) == itemSize);
            uint8_t *batchPtr = m_deviceBuffer + itemSize * i;
            checkCudaErrors(cudaMemcpy(batchPtr, from[i].data(),
                                       from[i].size() * sizeof(T),
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    /**
     * Fill a host buffer from the gradient buffer on the
     * device.
     *
     * Destination buffer will be resized if necessary.
     */
    void CopyGradBufferToHost(std::vector<T> &dst) {
        assert(m_deviceBufferGrad != nullptr);
        if (dst.size() != GetLinearSize()) {
            dst.resize(GetLinearSize());
        }
        checkCudaErrors(cudaMemcpy(dst.data(), m_deviceBufferGrad,
                                   GetSizeBytes(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    /**
     * Overloaded and custom Tensor Operations for Autodiff functionality
     */
    TensorPtr<T> Convolve(TensorPtr<T> filter, Stride2D stride = {1, 1},
                          Pad2D padding = {1, 1}) {
        return MakeConvolve(this->shared_from_this(), filter, stride, padding);
    }

    /**
     * Addition / Subtraction
     *
     * This function is Implemented with the TensorOp operation.
     */
    TensorPtr<T> Add(TensorPtr<T> rhs, T rhsMul = 1.0f);
    TensorPtr<T> Power(T scalar);

    friend TensorPtr<T> operator+(const TensorPtr<T> &lhs,
                                  const TensorPtr<T> &rhs) {
        TensorPtr<T> out = lhs->Add(rhs);
        return out;
    }

    friend TensorPtr<T> operator-(const TensorPtr<T> &lhs,
                                  const TensorPtr<T> &rhs) {
        TensorPtr<T> out = lhs->Add(rhs, -1);
        return out;
    }

    friend TensorPtr<T> operator^(TensorPtr<T> lhs, const T &rhs) {
        TensorPtr<T> out = lhs->Power(rhs);
        return out;
    }

    /**
     * Functions for loss calculations.
     */
    TensorPtr<T> Softmax();
    TensorPtr<T> ReLU();

    /**
     * Optimization
     */
    void ApplyGradient(const float stepSize) {
        assert(m_deviceBuffer != nullptr);
        assert(m_deviceBufferGrad != nullptr);
        // assert(m_filterDesc != nullptr);
        assert(m_tensorDesc != nullptr);
        cudnnOpTensorDescriptor_t tensorOpDesc;
        checkCudaErrors(cudnnCreateOpTensorDescriptor(&tensorOpDesc));
        std::array<float, 3> blendFactors{1.0, stepSize, 0.0};
        checkCudaErrors(
            cudnnSetOpTensorDescriptor(tensorOpDesc, CUDNN_OP_TENSOR_ADD,
                                       m_dataType, CUDNN_PROPAGATE_NAN));
        checkCudaErrors(cudnnOpTensor(
            GPUContext.GetCUDNNHandle(), tensorOpDesc, &blendFactors[0],
            GetTensorDesc(), GetDevicePointer(), &blendFactors[1],
            GetTensorDesc(), GetGradPointer(), &blendFactors[2],
            GetTensorDesc(), GetDevicePointer()));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudnnDestroyOpTensorDescriptor(tensorOpDesc));
    }

    /**
     * Dst will be resized it it is not large enough
     */
    template <typename DestinationType>
    void CopyBufferToHost(std::vector<DestinationType> &dst) {
        if (sizeof(T) == sizeof(DestinationType)) {
            // Case 1 : direct copy
            dst.resize(GetLinearSize());
            checkCudaErrors(cudaMemcpy(dst.data(), m_deviceBuffer,
                                       GetSizeBytes(), cudaMemcpyDeviceToHost));
        } else {
            // Case 2 : cast
            std::vector<T> tmp(GetLinearSize());
            checkCudaErrors(cudaMemcpy(tmp.data(), m_deviceBuffer,
                                       GetSizeBytes(), cudaMemcpyDeviceToHost));
            dst = std::vector<DestinationType>(tmp.begin(), tmp.end());
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

/**
 * Convenience Methods
 **/

template <typename T>
TensorPtr<T> CreateTensor(TensorShape shape, const std::string &name,
                          T constValueFill, bool grad = true) {
    TensorPtr<T> p = std::make_shared<Tensor<T>>();
    p->SetGradFlag(grad);
    p->SetShape(shape);
    p->SetName(name);
    p->AllocateIfNecessary();
    p->FillConstant(constValueFill);
    return p;
}

template <typename T> TensorPtr<T> CreateTensor() {
    TensorPtr<T> p = std::make_shared<Tensor<T>>();
    return p;
}

template <typename T>
TensorPtr<T> CreateFilter(int inChannel, int outChannel, int rows, int cols,
                          const std::string &name, T constValueFill,
                          bool grad = true) {
    TensorPtr<T> p = std::make_shared<Tensor<T>>();
    p->SetGradFlag(grad);
    p->SetFilterShape(inChannel, outChannel, rows, cols);
    p->SetName(name);
    p->AllocateIfNecessary();
    p->FillConstant(constValueFill);
    return p;
}

/**
 * AutoDiff
 *
 */
class AutoDiffContext {
  public:
    AutoDiffContext() {}
    ~AutoDiffContext() {}

    void AddOp(std::shared_ptr<BaseOperation> op) { m_opTrace.push_back(op); }
    void Reset() { m_opTrace.clear(); }
    unsigned int GetOpTraceSize() { return m_opTrace.size(); }

    void CalcGradient(TensorBasePtr scalarTensor,
                      std::vector<TensorBasePtr> &parameters) {
        LOG.INFO() << "Trainable parameters with names : ";
        for (auto p : parameters) {
            LOG.INFO() << p->GetName();
        }

        CalcGradient(scalarTensor);
    }

    void CalcGradient(TensorBasePtr scalarTensor) {
        LOG.DEBUG() << "Calc gradient of f'n with output name : "
                    << scalarTensor->GetName();
        // Initialize the backward operation. This operation sets up the
        // gradient tensor at the top of the chain.
        scalarTensor->InitGradChain();

        // Cycle through the operations in reverse order.
        for (auto opIter = m_opTrace.rbegin(); opIter != m_opTrace.rend();
             opIter++) {
            auto op = *opIter;
            LOG.DEBUG() << op->GetName();

            // Skip this op if it's output hasn't seen a backward pass.
            // this means that this op is somehow disconnected or upstream
            // from scalarTensor
            if (op->GetOutputTensor()->GetBackwardPasses() < 1) {
                LOG.DEBUG() << "Skipping op " << op->GetName();
                continue;
            }

            op->ExecuteBackward();
        }
        scalarTensor->ZeroBuffer();
    }

    /**
     * Performs simple SGD by applying the gradient.
     */
    void StepOptimizer(std::vector<TensorBasePtr> &params) {
        for (auto &t : params) {
            if (!t->GetGradFlag()) {
                throw DLFSError("Parameter does not require grad : " +
                                t->GetName());
            }
            LOG.DEBUG() << "Applying gradient to " << t->GetName();
            t->ApplyGradient(m_learningRate);
        }
    }

    /**
     * Prints out all information for debugging:
     * - Tensor and Op Traces
     * - Memory profile
     */
    std::string Print();

  private:
    /**
     * Optimizer settings
     * Should later be broken out into a seperate class
     */
    float m_learningRate{0.001};

  private:
    std::vector<BaseOpPtr> m_opTrace;
};

extern AutoDiffContext ADContext;

struct TensorInfoCUDA {
    int n;
    int h;
    int w;
    int c;

    TensorInfoCUDA(const TensorShape &shape) {
        n = shape[0];
        h = shape[1];
        w = shape[2];
        c = shape[3];
    }
};

} // namespace DLFS
