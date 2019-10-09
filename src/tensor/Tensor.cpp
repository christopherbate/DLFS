#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include "AutoDiff.hpp"
#include "Tensor.hpp"
#include "operations/Activation.hpp"
#include "operations/Convolution.hpp"
#include "operations/SigmoidCrossEntropy.hpp"
#include "operations/Softmax.hpp"
#include "operations/TensorOp.hpp"

using namespace DLFS;
using namespace std;

template <typename T> Tensor<T>::Tensor(const std::string &name) {
    m_shape = {0, 0, 0, 0};
    m_dataType = CUDNN_DATA_FLOAT;
    this->SetName(name);
}

template <typename T> Tensor<T>::~Tensor() {
    Deallocate();
    if (m_filterDesc)
        cudnnDestroyFilterDescriptor(m_filterDesc);
    if (m_dwFilterDesc)
        cudnnDestroyFilterDescriptor(m_dwFilterDesc);
    if (m_tensorDesc)
        cudnnDestroyTensorDescriptor(m_tensorDesc);
}

template <typename T> void Tensor<T>::SetShape(TensorShape shape) {
    m_isFilter = false;
    m_shape = shape;
    FillCUDNNDesc();
}

template <typename T> void Tensor<T>::SetShape(int b, int h, int w, int c) {
    m_isFilter = false;
    m_shape = std::array<int, 4>{b, h, w, c};
    FillCUDNNDesc();
}

template <typename T> std::string Tensor<T>::PrintTensor() {
    std::vector<T> buffer(this->GetLinearSize());
    std::ostringstream ss;
    CopyBufferToHost(buffer);
    ss.precision(2);

    // We loop over batches, then channels, printing the 2d arrays
    for (int j = 0; j < m_shape[0]; j++) {
        for (int ch = 0; ch < m_shape[3]; ch++) {
            ss << "["
               << "\n";
            for (int y = 0; y < m_shape[1]; y++) {
                for (int x = 0; x < m_shape[2]; x++) {
                    int idx = j * GetExpectedSizeWithinBatch() +
                              y * (m_shape[3] * m_shape[2]) + x * m_shape[3] +
                              ch;
                    ss << std::fixed << buffer[idx] << ", ";
                }
            }
            ss << "], \n";
        }
    }
    return ss.str();
}

/**
 * we use the NHWC filter shape,
 * which means the filters take the form of
 * KRSC (output, rows, cols, input)
 * **/
template <typename T>
void Tensor<T>::SetFilterShape(int numInputChn, int numOutputChn, int rows,
                               int cols) {
    m_isFilter = true;
    m_shape = {numOutputChn, rows, cols, numInputChn};
    FillCUDNNDesc();
}

/**
 * Sets up the tensor descriptor for CUDNN. Must be called after
 * the shape of the tensor changes.
 */
template <typename T> void Tensor<T>::FillCUDNNDesc() {
    if (m_isFilter) {
        if (!m_filterDesc)
            checkCudaErrors(cudnnCreateFilterDescriptor(&m_filterDesc));
        checkCudaErrors(cudnnSetFilter4dDescriptor(
            m_filterDesc, m_dataType, CUDNN_TENSOR_NHWC, m_shape[0], m_shape[3],
            m_shape[1], m_shape[2]));

        if (GetGradFlag()) {
            if (!m_dwFilterDesc)
                checkCudaErrors(cudnnCreateFilterDescriptor(&m_dwFilterDesc));
            checkCudaErrors(cudnnSetFilter4dDescriptor(
                m_dwFilterDesc, m_dataType, CUDNN_TENSOR_NHWC, m_shape[0],
                m_shape[3], m_shape[1], m_shape[2]));
        }
    } else {
        if (!m_tensorDesc)
            checkCudaErrors(cudnnCreateTensorDescriptor(&m_tensorDesc));
        checkCudaErrors(cudnnSetTensor4dDescriptor(
            m_tensorDesc, CUDNN_TENSOR_NHWC, m_dataType, m_shape[0], m_shape[3],
            m_shape[1], m_shape[2]));
    }
}

template <typename T> void Tensor<T>::AllocateIfNecessary() {
    size_t needed_bytes = GetLinearSize() * sizeof(T);
    bool needGradBuffer = GetGradFlag() && m_deviceBufferGrad == nullptr;
    if (needed_bytes > m_bufferSize || m_deviceBuffer == nullptr ||
        needGradBuffer) {
        LOG.DEBUG() << "Needed bytes:" << needed_bytes
                    << " current allocation: " << m_bufferSize;
        Allocate();
    }
}

template <typename T> void Tensor<T>::Allocate() {
    if (m_deviceBuffer != nullptr || m_deviceBufferGrad != nullptr)
        Deallocate();

    m_bufferSize = GetLinearSize() * sizeof(T);

    LOG.DEBUG() << "Allocating tensor " << GetName() << ":" << GetId() << " "
                << (float)m_bufferSize / 1024.0 << " KB";
    checkCudaErrors(cudaMalloc(&m_deviceBuffer, m_bufferSize));

    if (GetGradFlag()) {
        LOG.DEBUG() << "Allocating grad tensor " << GetName() << ":" << GetId()
                    << " " << (float)m_bufferSize / 1024.0 << " KB";
        checkCudaErrors(cudaMalloc(&m_deviceBufferGrad, m_bufferSize));
    }
}

template <typename T> void Tensor<T>::Deallocate() {
    if (m_deviceBuffer != nullptr) {
        LOG.DEBUG() << "Attempting to deallocate Tensor " << GetName() << ":"
                    << GetId();
        checkCudaErrors(cudaFree(m_deviceBuffer));
        if (m_deviceBufferGrad == m_deviceBuffer)
            m_deviceBufferGrad = nullptr;
        m_deviceBuffer = nullptr;
        m_bufferSize = 0;
    }
    if (m_deviceBufferGrad != nullptr) {
        LOG.DEBUG() << "Attempting to deallocate grad Tensor " << GetName()
                    << ":" << GetId();
        checkCudaErrors(cudaFree(m_deviceBufferGrad));
        m_deviceBufferGrad = nullptr;
    }
}

template <typename T>
TensorPtr<T> Tensor<T>::Convolve(TensorPtr<T> filter, Pad2d padding,
                                 Stride2d stride) {
    shared_ptr<Convolution<T>> convOp =
        make_shared<Convolution<T>>(padding, stride);

    convOp->SetFilter(filter);
    convOp->SetFeatures(this->shared_from_this());

    // Create the output tensor.
    TensorShape outputShape = convOp->Prepare();
    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();

    // Put this variable into the active set if either inputs are active
    if (filter->GetGradFlag() || GetGradFlag())
        outputTensor->SetGradFlag(true);

    outputTensor->SetName("ConvOutput");
    outputTensor->SetShape(outputShape);
    outputTensor->AllocateIfNecessary();

    convOp->SetOutput(outputTensor);

    LOG << "Conv op prepared, output shape: " << outputTensor->PrintShape();

    convOp->ExecuteForward();

    LOG << "Executed ";

    ADContext.AddOp(convOp);

    return outputTensor;
}

template <typename T> TensorPtr<T> Tensor<T>::Add(TensorPtr<T> rhs) {
    shared_ptr<TensorOp<T>> addOp = make_shared<TensorOp<T>>(PW_ADD);

    addOp->SetName("AddOp");
    addOp->SetInput(this->shared_from_this(), 0);
    addOp->SetInput(rhs, 1);

    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();
    if (rhs->GetGradFlag() || GetGradFlag())
        outputTensor->SetGradFlag(true);
    outputTensor->SetShape(GetShape());
    outputTensor->SetName("AddOutput");
    outputTensor->AllocateIfNecessary();

    addOp->SetOutput(outputTensor);

    addOp->ExecuteForward();

    ADContext.AddOp(addOp);

    return outputTensor;
}

template <typename T> TensorPtr<T> Tensor<T>::Power(T scalar) {
    shared_ptr<TensorOp<T>> powOp = make_shared<TensorOp<T>>(PW_POW);

    powOp->SetName("PowerOp");
    powOp->SetInput(this->shared_from_this(), 0);
    powOp->SetPower(scalar);

    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();
    if (GetGradFlag())
        outputTensor->SetGradFlag(true);
    outputTensor->SetShape(GetShape());
    outputTensor->SetName("PowerOutput");
    outputTensor->AllocateIfNecessary();

    powOp->SetOutput(outputTensor);

    powOp->ExecuteForward();

    ADContext.AddOp(powOp);

    return outputTensor;
}

template <typename T> TensorPtr<T> Tensor<T>::Softmax() {
    SoftmaxOpPtr<T> op = std::make_shared<SoftmaxOp<T>>();

    op->SetName("SoftmaxOp");
    op->SetInput(this->shared_from_this());

    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();
    if (GetGradFlag())
        outputTensor->SetGradFlag(true);
    outputTensor->SetShape(GetShape());
    outputTensor->SetName("PowerOutput");
    outputTensor->AllocateIfNecessary();

    op->SetOutput(outputTensor);

    op->ExecuteForward();

    ADContext.AddOp(op);

    return outputTensor;
}

template <typename T> TensorPtr<T> Tensor<T>::ReLU() {
    ActivationOpPtr<T> op = std::make_shared<ActivationOp<T>>();

    op->SetName("ReluOp");
    op->SetInput(this->shared_from_this());

    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();
    if (GetGradFlag())
        outputTensor->SetGradFlag(true);

    outputTensor->SetShape(GetShape());
    outputTensor->SetName("ReluOutput");
    outputTensor->AllocateIfNecessary();

    op->SetOutput(outputTensor);

    op->ExecuteForward();

    ADContext.AddOp(op);

    return outputTensor;
}

template <typename T>
TensorPtr<T> Tensor<T>::SigmoidCELoss(TensorPtr<uint16_t> labels) {
    SigmoidCEOpPtr<T> op = std::make_shared<SigmoidCrossEntropyOp<T>>();

    op->SetName("SigmoidCEOp");
    op->SetLogits(this->shared_from_this());
    op->SetLabels(labels);

    TensorPtr<T> outputTensor = ADContext.CreateTensor<T>();
    if (GetGradFlag())
        outputTensor->SetGradFlag(true);

    outputTensor->SetShape(GetShape());
    outputTensor->SetName("SigmoidCELossValue");
    outputTensor->AllocateIfNecessary();

    op->SetOutput(outputTensor);

    op->ExecuteForward();

    ADContext.AddOp(op);

    return outputTensor;
}

template <typename T>
template <typename TargetType>
TensorPtr<TargetType> Tensor<T>::Cast() {
    TensorPtr<TargetType> cTensor = ADContext.CreateTensor<TargetType>();
    cTensor->SetGradFlag(this->m_calcGrad);
    cTensor->SetShape(this->m_shape);
    cTensor->SetName(this->m_name + "-cast");
    cTensor->AllocateIfNecessary();

    // Perform conversion on the host side (should update to
    // CUDA kernel soon).
    std::vector<T> buffer(GetLinearSize());
    std::vector<TargetType> newBuffer(GetLinearSize());

    this->CopyBufferToHost(buffer);
    for (unsigned int i = 0; i < buffer.size(); i++) {
        newBuffer[i] = static_cast<TargetType>(buffer[i]);
    }
    cTensor->CopyBufferToDevice(newBuffer);

    return cTensor;
}

template class Tensor<float>;
template class Tensor<uint8_t>;
template class Tensor<uint16_t>;

template TensorPtr<float> Tensor<uint8_t>::Cast();
template TensorPtr<uint8_t> Tensor<float>::Cast();