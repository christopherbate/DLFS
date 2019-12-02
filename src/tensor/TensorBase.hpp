#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cudnn.h>
#include <memory>
#include <string>

#include "Logging.hpp"

namespace DLFS {

using TensorShape = std::array<int, 4>;
using FilterShape = std::array<int, 4>;

// Types
// Float16 indicated device side should be float16.
// host side values are floats.
using float16 = float;

/**
 * TensorBase is the base class
 * for the Tensor object.
 *
 * We need this interface for the autograd and other objects
 * that need to act on Tensors with multipel types
 */

class TensorBase {
  public:
    // the default data type is float
    TensorBase() : m_dataType{CUDNN_DATA_FLOAT} {}
    virtual ~TensorBase() { Deallocate(); }

    /**
     * Implementation required
     **/
    virtual void InitGradChain() = 0;
    virtual void ApplyGradient(const float step) = 0;
    virtual const std::string &PrintTensor(bool grad, bool breakChannels) = 0;
    virtual size_t GetDataTypeSize() = 0;

    /**
     * Implementation not-required
     **/
    std::array<int, 4> GetShape() const { return m_shape; }

    void SetShape(const TensorShape &shape) {
        m_isFilter = false;
        m_shape = shape;
        FillCUDNNDesc();
    };

    void SetShape(int n, int h, int w, int c) {
        m_isFilter = false;
        m_shape = std::array<int, 4>{n, h, w, c};
        FillCUDNNDesc();
    }

    size_t GetLinearSize() {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]);
    }

    uint8_t *GetDevicePointer() { return (uint8_t *)m_deviceBuffer; }

    uint8_t *GetGradPointer() { return m_deviceBufferGrad; }

    /**
     * Returns size of tensor in bytes
     */
    size_t GetSizeBytes() {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]) *
               GetDataTypeSize();
    }

    /**
     * we use the NHWC filter shape,
     * which means the filters take the form of
     * KRSC ( #out_chs, rows, cols, # in_chs)
     * **/
    void SetFilterShape(int numInputChn, int numOutputChn, int rows, int cols) {
        m_isFilter = true;
        m_shape = {numOutputChn, rows, cols, numInputChn};
        FillCUDNNDesc();
    }
    void SetFilterShape(const FilterShape &shape) {
        SetFilterShape(shape[0], shape[1], shape[2], shape[3]);
    }

    const std::string &GetName() { return m_name; }
    void SetName(const std::string &name) { m_name = name; }

    void SetGradFlag(bool shouldCalcGrad) { m_calcGrad = shouldCalcGrad; }

    bool GetGradFlag() { return m_calcGrad; }

    void ResetBackwardPasses() { m_numBackPasses = 0; }
    uint32_t GetBackwardPasses() { return m_numBackPasses; }
    bool IsFirstBackwardPass() { return m_numBackPasses == 0; }

    void IncrementBackwardPass() { m_numBackPasses++; }

    const std::string &PrintShape() {
        m_shapeMsg = "(";
        for (auto d : m_shape)
            m_shapeMsg += std::to_string(d) + ", ";
        m_shapeMsg += ")";
        return m_shapeMsg;
    }

    cudnnTensorDescriptor_t GetTensorDesc() { return m_tensorDesc; }
    cudnnFilterDescriptor_t GetFilterDesc() { return m_filterDesc; }

    void AllocateIfNecessary() {
        size_t needed_bytes = GetSizeBytes();
        bool needGradBuffer = GetGradFlag() && m_deviceBufferGrad == nullptr;
        if (needed_bytes > m_bufferSize || m_deviceBuffer == nullptr ||
            needGradBuffer) {
            LOG.DEBUG() << "Needed bytes:" << needed_bytes
                        << " current allocation: " << m_bufferSize;
            Allocate();
        }
    }

  public:
    /**
     * Retrieves a vector of device-side buffer pointers
     * which each point to an individual sub-tensor in order based
     * on the first dimension of this tensor.
     */
    std::vector<uint8_t *> GetIterablePointersOverBatch() {
        std::vector<uint8_t *> batchPointers(m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3] * GetDataTypeSize();
        for (int i = 0; i < m_shape[0]; i++) {
            batchPointers[i] = m_deviceBuffer + itemSize * i;
        }
        return batchPointers;
    }

    void ZeroGrad() {
        assert(m_deviceBufferGrad != nullptr);
        checkCudaErrors(cudaMemset(m_deviceBufferGrad, 0, GetSizeBytes()));
    }

    void ZeroBuffer() {
        assert(m_deviceBuffer != nullptr);
        checkCudaErrors(cudaMemset(m_deviceBuffer, 0, GetSizeBytes()));
    }

  protected:
    cudnnDataType_t m_dataType;
    TensorShape m_shape;
    bool m_calcGrad{false};
    std::string m_name{"UnnamedTensor"};

    /* How many times this tensor was seen as
       an input into an op during backward pass */
    uint32_t m_numBackPasses{0};

    // Buffers owned by every Tensor
    uint8_t *m_deviceBuffer{nullptr};
    uint8_t *m_deviceBufferGrad{nullptr};

    // CUDNN Tensor Descriptors
    cudnnTensorDescriptor_t m_tensorDesc{nullptr};
    cudnnFilterDescriptor_t m_filterDesc{nullptr};

    // Cached strings for printing
    std::string m_shapeMsg;
    std::string m_tensorMsg;

    size_t m_bufferSize{0};

    void Allocate() {
        if (m_deviceBuffer != nullptr || m_deviceBufferGrad != nullptr)
            Deallocate();

        m_bufferSize = GetSizeBytes();

        LOG.DEBUG() << "Allocating tensor " << GetName() << " "
                    << (float)m_bufferSize / 1024.0 << " KB";
        checkCudaErrors(cudaMalloc(&m_deviceBuffer, m_bufferSize));

        if (GetGradFlag()) {
            LOG.DEBUG() << "Allocating grad tensor " << GetName() << " "
                        << (float)m_bufferSize / 1024.0 << " KB";
            checkCudaErrors(cudaMalloc(&m_deviceBufferGrad, m_bufferSize));
        }
    }

    void Deallocate() {
        if (m_deviceBuffer != nullptr) {
            LOG.DEBUG() << "Attempting to deallocate Tensor " << GetName();
            checkCudaErrors(cudaFree(m_deviceBuffer));
            if (m_deviceBufferGrad == m_deviceBuffer)
                m_deviceBufferGrad = nullptr;
            m_deviceBuffer = nullptr;
            m_bufferSize = 0;
        }
        if (m_deviceBufferGrad != nullptr) {
            LOG.DEBUG() << "Attempting to deallocate grad Tensor " << GetName();
            checkCudaErrors(cudaFree(m_deviceBufferGrad));
            m_deviceBufferGrad = nullptr;
        }
    }

  private:
    void FillCUDNNDesc() {
        // Only create the filter description where necessary
        if (!m_filterDesc && m_isFilter) {
            checkCudaErrors(cudnnCreateFilterDescriptor(&m_filterDesc));
            checkCudaErrors(cudnnSetFilter4dDescriptor(
                m_filterDesc, m_dataType, CUDNN_TENSOR_NHWC, m_shape[0],
                m_shape[3], m_shape[1], m_shape[2]));
        }

        // Add tensors, including filters, need tensor descriptor.
        if (!m_tensorDesc) {
            checkCudaErrors(cudnnCreateTensorDescriptor(&m_tensorDesc));
            checkCudaErrors(cudnnSetTensor4dDescriptor(
                m_tensorDesc, CUDNN_TENSOR_NHWC, m_dataType, m_shape[0],
                m_shape[3], m_shape[1], m_shape[2]));
        }
    }
    bool m_isFilter{false};
}; // namespace DLFS

using TensorBasePtr = std::shared_ptr<TensorBase>;

} // namespace DLFS