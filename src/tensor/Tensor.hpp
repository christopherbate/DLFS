#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "operations/OpsCommon.hpp"
#include "../Logging.hpp"

namespace DLFS
{

typedef std::array<int, 4> TensorShape;

class TensorBase
{
public:
    virtual void SetShape(TensorShape shape) = 0;

    inline std::string GetName()
    {
        return m_name;
    }

    inline void SetName(const std::string &name)
    {
        m_name = name;
    }

    inline void SetId(uint32_t id)
    {
        m_id = id;
    }

    inline uint32_t GetId()
    {
        return m_id;
    }

private:
    uint32_t m_id{0};
    std::string m_name{"UnnamedTensor"};
};

using TensorBasePtr = std::shared_ptr<TensorBase>;

/*
 Fwd Declare so we can use the Tensor Ptr type in the 
class decl. 
*/
template <typename T>
class Tensor;

template <typename T>
using TensorPtr = std::shared_ptr<Tensor<T>>;

/**
 * Tensor represents an array of double/float/half/int numbers.
 */
template <typename T>
class Tensor : public TensorBase, public std::enable_shared_from_this<Tensor<T>>
{
public:
    Tensor();
    ~Tensor();

    void SetShape(TensorShape shape);
    void SetShape(int b, int h, int w, int c);
    void SetFilterShape(int numInputChn,
                        int numOutputChn,
                        int rows, int cols);
    void FillCUDNNDesc();

    void AllocateIfNecessary();

    /**
    * Initialization Functions
    */

    /* Fill buffer with constant value */
    void FillConstant(T constVal)
    {
        assert(m_deviceBuffer != nullptr);

        std::vector<T> localBuffer(GetLinearSize(), constVal);
        checkCudaErrors(cudaMemcpy(m_deviceBuffer, localBuffer.data(),
                                   localBuffer.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    /* Fill gradient buffer with constant value */
    void FillConstantGrad(T constVal)
    {
        assert(m_deviceBufferGrad != nullptr);
        std::vector<T> localBuffer(GetLinearSize(), constVal);
        checkCudaErrors(cudaMemcpy(m_deviceBufferGrad, localBuffer.data(),
                                   localBuffer.size() * sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    /* Fill buffer with constant value */
    TensorPtr<T> Convolve(TensorPtr<T> filter, Pad2d padding = {1, 1},
                          Stride2d = {1, 1});

    inline std::vector<unsigned char *> GetIterablePointersOverBatch()
    {
        std::vector<unsigned char *> batchPointers(m_shape[0]);
        int itemSize = m_shape[1] * m_shape[2] * m_shape[3];
        for (int i = 0; i < m_shape[0]; i++)
        {
            batchPointers[i] = m_deviceBuffer + itemSize * i;
        }
        return batchPointers;
    }

    inline unsigned char *GetPointer()
    {
        return (unsigned char *)m_deviceBuffer;
    }

    inline std::array<int, 4> GetShape() const
    {
        return m_shape;
    }

    inline size_t GetExpectedSize()
    {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline size_t GetLinearSize()
    {
        return (m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3]);
    }

    inline size_t GetExpectedSizeWithinBatch()
    {
        return (m_shape[1] * m_shape[2] * m_shape[3]) * m_pitch;
    }

    inline cudnnTensorDescriptor_t GetTensorDesc()
    {
        return m_tensorDesc;
    }

    inline cudnnFilterDescriptor_t GetFilterDesc()
    {
        return m_filterDesc;
    }

    inline std::string PrintShape()
    {
        std::string shapeMsg = "(";
        for (auto d : m_shape)
            shapeMsg += std::to_string(d) + ", ";
        shapeMsg += ")";
        return shapeMsg;
    }

    inline size_t GetPitch()
    {
        return m_pitch;
    }

    inline cudnnFilterDescriptor_t GetGradFilterDesc()
    {
        return m_dwFilterDesc;
    }

    inline void SetGradFlag(bool shouldCalcGrad)
    {
        m_calcGrad = shouldCalcGrad;
    }

    inline bool GetGradFlag()
    {
        return m_calcGrad;
    }

    inline uint8_t *GetGradPointer()
    {
        return m_deviceBufferGrad;
    }

    inline void InitGradChain()
    {
        assert(m_deviceBuffer != nullptr);

        if(m_deviceBufferGrad){
            checkCudaErrors(cudaFree(m_deviceBufferGrad));            
        }
        m_deviceBufferGrad = m_deviceBuffer;
    }
    
    void CopyBufferToHost(std::vector<T> &dst)
    {
        if (dst.size() != GetLinearSize())
        {
            dst.resize(GetLinearSize());
        }
        cudaMemcpy(dst.data(), m_deviceBuffer,
                   GetLinearSize() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

    void CopyGradBufferToHost(std::vector<T> &dst)
    {
        if (dst.size() != GetLinearSize())
        {
            dst.resize(GetLinearSize());
        }
        cudaMemcpy(dst.data(), m_deviceBufferGrad,
                   GetLinearSize() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

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
    bool m_calcGrad{false};

private:
    void Allocate();
    void Deallocate();
};

} // namespace DLFS
