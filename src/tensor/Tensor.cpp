#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

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

    if (std::is_same<T, float>::value) {
        m_dataType = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, double>::value) {
        m_dataType = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, uint8_t>::value) {
        m_dataType = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, uint32_t>::value) {
        m_dataType = CUDNN_DATA_INT32;
    } else {
        throw DLFSError("Unsupported data type");
    }
    this->SetName(name);
}

template <typename T> Tensor<T>::~Tensor() {
    if (m_filterDesc)
        checkCudaErrors(cudnnDestroyFilterDescriptor(m_filterDesc));
    if (m_tensorDesc)
        checkCudaErrors(cudnnDestroyTensorDescriptor(m_tensorDesc));
}

template <typename T>
const std::string &Tensor<T>::PrintTensor(bool grad, bool breakChannels) {
    std::vector<T> buffer(this->GetLinearSize());
    std::ostringstream ss;
    if (!grad) {
        CopyBufferToHost(buffer);
    } else {
        CopyGradBufferToHost(buffer);
    }

    ss.precision(3);

    int imgArea = m_shape[1] * m_shape[2] * m_shape[3];
    int rowArea = m_shape[2] * m_shape[3];

    // We loop over batches, then channels, printing the 2d arrays
    ss << "\n";
    for (int j = 0; j < m_shape[0]; j++) {
        for (int ch = 0; ch < m_shape[3]; ch++) {
            ss << "[";

            if (breakChannels)
                ss << "\n";

            for (int y = 0; y < m_shape[1]; y++) {
                ss << "[";
                for (int x = 0; x < m_shape[2]; x++) {
                    int idx = j * imgArea + y * (rowArea) + x * m_shape[3] + ch;
                    ss << std::fixed << buffer[idx] << ",";
                }
                ss << "]";
                if (breakChannels) {
                    ss << "\n";
                }
            }
            ss << "]";
            if (breakChannels) {
                ss << "\n";
            }
        }

        ss << "\n";
    }
    m_tensorMsg = ss.str();
    return m_tensorMsg;
}


template <typename T> TensorPtr<T> Tensor<T>::Add(TensorPtr<T> rhs, T rhsMul) {
    shared_ptr<TensorOp<T>> addOp = make_shared<TensorOp<T>>(PW_ADD);

    addOp->SetName("AddOp");
    addOp->SetInput(this->shared_from_this(), 0);
    addOp->SetInput(rhs, 1);

    TensorPtr<T> outputTensor = CreateTensor<T>();
    if (rhs->GetGradFlag() || GetGradFlag())
        outputTensor->SetGradFlag(true);
    outputTensor->SetShape(GetShape());
    outputTensor->SetName("AddOutput");
    outputTensor->AllocateIfNecessary();

    addOp->SetOutput(outputTensor);
    addOp->SetRHSScale(rhsMul);
    addOp->ExecuteForward();

    ADContext.AddOp(addOp);

    return outputTensor;
}

template <typename T> TensorPtr<T> Tensor<T>::Power(T scalar) {
    shared_ptr<TensorOp<T>> powOp = make_shared<TensorOp<T>>(PW_POW);

    powOp->SetName("PowerOp");
    powOp->SetInput(this->shared_from_this(), 0);
    powOp->SetPower(scalar);

    TensorPtr<T> outputTensor = CreateTensor<T>();
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

    TensorPtr<T> outputTensor = CreateTensor<T>();
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

    TensorPtr<T> outputTensor = CreateTensor<T>();
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
TensorPtr<T> Tensor<T>::SigmoidCELoss(TensorPtr<uint32_t> labels) {
    SigmoidCEOpPtr<T> op = std::make_shared<SigmoidCrossEntropyOp<T>>();

    op->SetName("SigmoidCEOp");
    op->SetLogits(this->shared_from_this());
    op->SetLabels(labels);

    TensorPtr<T> outputTensor = CreateTensor<T>();
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
    TensorPtr<TargetType> cTensor = CreateTensor<TargetType>();
    cTensor->SetGradFlag(this->m_calcGrad);
    cTensor->SetShape(this->m_shape);
    cTensor->SetName(this->m_name + "-cast");
    cTensor->AllocateIfNecessary();

    // Perform conversion on the host side (should update to
    // CUDA kernel).
    std::vector<T> buffer(GetLinearSize());
    std::vector<TargetType> newBuffer(GetLinearSize());

    this->CopyBufferToHost(buffer);
    for (unsigned int i = 0; i < buffer.size(); i++) {
        newBuffer[i] = static_cast<TargetType>(buffer[i]);
    }
    cTensor->CopyBufferToDevice(newBuffer);

    return cTensor;
}

template class DLFS::Tensor<float>;
template class DLFS::Tensor<uint8_t>;
template class DLFS::Tensor<uint32_t>;

template DLFS::TensorPtr<float> DLFS::Tensor<uint8_t>::Cast();
template DLFS::TensorPtr<uint8_t> DLFS::Tensor<float>::Cast();