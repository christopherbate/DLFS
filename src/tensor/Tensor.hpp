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
#include "operations/types.hpp"

namespace DLFS {

typedef std::array<int, 4> TensorShape;

class TensorBase {
  public:
    virtual void SetShape(TensorShape shape) = 0;
    virtual void InitGradChain() = 0;

    inline std::string GetName() { return m_name; }

    inline void SetName(const std::string &name) { m_name = name; }

    inline void SetId(uint32_t id) { m_id = id; }

    inline uint32_t GetId() { return m_id; }

    inline void SetGradFlag(bool shouldCalcGrad) {
        m_calcGrad = shouldCalcGrad;
    }

    inline bool GetGradFlag() { return m_calcGrad; }

    inline void ResetBackwardPasses() { m_numBackPasses = 0; }

    inline uint32_t GetBackwardPasses() { return m_numBackPasses; }

    inline bool IsFirstBackwardPass() { return m_numBackPasses == 0; }

    inline void IncrementBackwardPass() { m_numBackPasses++; }

  protected:
    bool m_calcGrad{false};
    uint32_t m_id{0};
    std::string m_name{"UnnamedTensor"};

    /* How many times this tensor was seen as
       an input into an op during backward pass */
    uint32_t m_numBackPasses{0};
};

using TensorBasePtr = std::shared_ptr<TensorBase>;

/*
 Fwd Declare so we can use the Tensor Ptr type in the
class decl.
*/
template <typename T> class Tensor;

template <typename T> using TensorPtr = std::shared_ptr<Tensor<T>>;

/**
 * Tensor represents an array of double/float/half/int numbers.
 */
template <typename T>
class Tensor : public TensorBase,
               public std::enable_shared_from_this<Tensor<T>> {
  public:
    Tensor(const std::string &name = "");
    ~Tensor();

    void SetShape(TensorShape shape);
    void SetShape(int b, int h, int w, int c);
    void SetFilterShape(int numInputChn, int numOutputChn, int rows, int cols);
    void FillCUDNNDesc();

    void AllocateIfNecessary();

    /**
     * Type-Casting
     * Returns a pointer to a new Tensor with the new type.
     *
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

    /**
     * Retrieves a vector of device-side buffer pointers
     * which each point to an individual sub-tensor in order based
     * on the first dimension of this tensor.
     */
    inline std::vector<unsigned char *> GetIterablePointersOverBatch() {
        std::vector<unsigned char *> batchPointers(m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3] * sizeof(T);
        for (int i = 0; i < m_shape[0]; i++) {
            batchPointers[i] = m_deviceBuffer + itemSize * i;
        }
        return batchPointers;
    }

    /**
     * Returns device pointer
     *
     * No gauruntees on its validity.
     */
    inline unsigned char *GetDevicePointer() {
        return (unsigned char *)m_deviceBuffer;
    }

    inline std::array<int, 4> GetShape() const { return m_shape; }

    inline size_t GetExpectedSize() {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline size_t GetLinearSize() {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]);
    }

    inline size_t GetExpectedSizeWithinBatch() {
        return (m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline cudnnTensorDescriptor_t GetTensorDesc() { return m_tensorDesc; }

    inline cudnnFilterDescriptor_t GetFilterDesc() { return m_filterDesc; }

    inline std::string PrintShape() {
        std::string shapeMsg = "(";
        for (auto d : m_shape)
            shapeMsg += std::to_string(d) + ", ";
        shapeMsg += ")";
        return shapeMsg;
    }

    std::string PrintTensor();

    inline size_t GetPitch() { return m_pitch; }

    inline cudnnFilterDescriptor_t GetGradFilterDesc() {
        return m_dwFilterDesc;
    }

    inline uint8_t *GetGradPointer() { return m_deviceBufferGrad; }

    inline void InitGradChain() {
        assert(m_deviceBuffer != nullptr);
        FillConstantGrad(1.0f);
        ResetBackwardPasses();
        IncrementBackwardPass();
    }

    /**
     * Dst will be resized it it is not large enough
     */
    inline void CopyBufferToHost(std::vector<T> &dst) {
        if (dst.size() != GetLinearSize()) {
            dst.resize(GetLinearSize());
        }

        assert(dst.size() == GetLinearSize());

        cudaMemcpy(dst.data(), m_deviceBuffer, GetLinearSize() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

    /**
     * Buffer from must be the correct size.
     * device buffer must already have been allocated.
     */
    inline void CopyBufferToDevice(const std::vector<T> &from) {
        assert(from.size() * sizeof(T) == GetLinearSize() * sizeof(T));
        assert(m_deviceBuffer != nullptr);
        checkCudaErrors(cudaMemcpy(m_deviceBuffer, from.data(),
                                   GetLinearSize() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    /**
     * Accepts a vector of (host) buffers and copies them to the host buffer
     * along the batch dimension.
     *
     * The individual buffers from the host all need to be the size of the
     * sub-batch slice on the device.
     */
    inline void CopyBatchBuffersToDevice(std::vector<std::vector<T>> &from) {
        assert((int)from.size() == m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3] * sizeof(T);
        for (int i = 0; i < m_shape[0]; i++) {
            assert((int)(from[i].size() * sizeof(T)) == itemSize);
            uint8_t *batchPtr = m_deviceBuffer + itemSize * i;
            checkCudaErrors(cudaMemcpy(batchPtr, from[i].data(),
                                       from[i].size() * sizeof(T),
                                       cudaMemcpyHostToDevice));
        }
    }

    /**
     * Faill a host buffer from the gradient buffer on the
     * device.
     *
     * Destination buffer will be resized if necessary.
     */
    void CopyGradBufferToHost(std::vector<T> &dst) {
        assert(m_deviceBufferGrad != nullptr);
        if (dst.size() != GetLinearSize()) {
            dst.resize(GetLinearSize());
        }
        cudaMemcpy(dst.data(), m_deviceBufferGrad, GetLinearSize() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

    /**
     * Overloaded and custom Tensor Operations for Autodiff functionality
     */
    TensorPtr<T> Convolve(TensorPtr<T> filter, Pad2d padding = {1, 1},
                          Stride2d = {1, 1});

    /**
     * Addition / Subtraction
     *
     * This function is Implemented with the TensorOp operation.
     */
    TensorPtr<T> Add(TensorPtr<T> rhs);
    TensorPtr<T> Power(T scalar);

    friend TensorPtr<T> operator+(const TensorPtr<T> &lhs,
                                  const TensorPtr<T> &rhs) {
        TensorPtr<T> out = lhs->Add(rhs);
        return out;
    }

    friend TensorPtr<T> operator-(const TensorPtr<T> &lhs,
                                  const TensorPtr<T> &rhs) {
        TensorPtr<T> out = lhs->Add(rhs);
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
    TensorPtr<T> SigmoidCELoss(TensorPtr<uint16_t> labels);
    TensorPtr<T> ReLU();

  private:
    TensorShape m_shape;

    cudnnDataType_t m_dataType;

    cudnnTensorDescriptor_t m_tensorDesc{nullptr};

    cudnnFilterDescriptor_t m_filterDesc{nullptr};
    cudnnFilterDescriptor_t m_dwFilterDesc{nullptr};

    size_t m_bufferSize{0};

    uint8_t *m_deviceBuffer{nullptr};
    uint8_t *m_deviceBufferGrad{nullptr};

    size_t m_pitch{sizeof(T)};

    bool m_isFilter{false};

  private:
    void Allocate();
    void Deallocate();
};

} // namespace DLFS
