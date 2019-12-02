#pragma once

#include "operations/BaseOperation.hpp"
#include "operations/types.hpp"
#include "tensor/Tensor.hpp"

#include <array>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

namespace DLFS {

template <typename FeatureDataType, typename FilterDataType>
class Convolution : public BaseOperation {
  public:
    Convolution(TensorPtr<FeatureDataType> features,
                TensorPtr<FilterDataType> filters, Pad2D padding,
                Stride2D stride)
        : m_features(features), m_filter(filters), m_bias(nullptr),
          m_output(nullptr) {
        Reset();
        m_strides = stride;
        m_padding = padding;
    }

    ~Convolution() {
        checkCudaErrors(cudnnDestroyConvolutionDescriptor(m_convDesc));
    }

    TensorShape Prepare() {
        assert(m_features != nullptr);
        assert(m_filter != nullptr);
        assert(m_convDesc != nullptr);

        checkCudaErrors(cudnnSetConvolution2dDescriptor(
            m_convDesc, m_padding[0], m_padding[1], m_strides[0], m_strides[1],
            1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        cudnnSetConvolutionMathType(m_convDesc, CUDNN_DEFAULT_MATH);

        TensorShape outputDims;
        checkCudaErrors(cudnnGetConvolution2dForwardOutputDim(
            m_convDesc, m_features->GetTensorDesc(), m_filter->GetFilterDesc(),
            &outputDims[0], &outputDims[3], &outputDims[1], &outputDims[2]));

        m_output =
            CreateTensor<FeatureDataType>(outputDims, "ConvOutput", 0.0, true);
        return outputDims;
    }

    TensorPtr<FeatureDataType> ExecuteForward() {
        Prepare();

        assert(m_workspaceBuffer == NULL);
        assert(m_workspaceSize == 0);
        assert(m_features->GetDevicePointer() != nullptr);
        assert(m_filter->GetDevicePointer() != nullptr);

        size_t wsSize = 0;

        // TODO: this should only be done when dimensions / algorithm change.
        cudnnGetConvolutionForwardWorkspaceSize(
            GPUContext.GetCUDNNHandle(), m_features->GetTensorDesc(),
            m_filter->GetFilterDesc(), m_convDesc, m_output->GetTensorDesc(),
            m_convFwdAlg, &wsSize);

        LOG.DEBUG() << "Executing cudnn ConvFwd kernel, "
                    << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

        unsigned char *devWs = NULL;
        if (wsSize > 0)
            checkCudaErrors(cudaMalloc(&devWs, wsSize));

        checkCudaErrors(cudnnConvolutionForward(
            GPUContext.GetCUDNNHandle(), &m_scaling[0],
            m_features->GetTensorDesc(), m_features->GetDevicePointer(),
            m_filter->GetFilterDesc(), m_filter->GetDevicePointer(), m_convDesc,
            m_convFwdAlg, devWs, wsSize, &m_scaling[1],
            m_output->GetTensorDesc(), m_output->GetDevicePointer()));
        if (wsSize > 0)
            checkCudaErrors(cudaFree(devWs));

        cudaDeviceSynchronize();

        return m_output;
    }

    void ExecuteBackward() {
        assert(m_features->GetDevicePointer() != nullptr);
        assert(m_filter->GetDevicePointer() != nullptr);

        size_t wsSize = 0;
        unsigned char *devWs = NULL;

        FeatureDataType blendFactors[2] = {1, 1};

        // Get grad with respect to filter.
        if (m_filter->GetGradFlag()) {
            blendFactors[1] = m_filter->IsFirstBackwardPass() ? 0 : 1;
            m_filter->IncrementBackwardPass();

            // TODO: this should only be done when dimensions / algorithm
            // change.
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                GPUContext.GetCUDNNHandle(), m_features->GetTensorDesc(),
                m_output->GetTensorDesc(), m_convDesc,
                m_filter->GetFilterDesc(), m_convBwdFilterAlg, &wsSize);

            LOG.DEBUG() << "Execuing cudnn ConvBwdFilter kernel "
                        << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

            if (wsSize > 0)
                checkCudaErrors(cudaMalloc(&devWs, wsSize));

            checkCudaErrors(cudnnConvolutionBackwardFilter(
                GPUContext.GetCUDNNHandle(), &blendFactors[0],
                m_features->GetTensorDesc(), m_features->GetDevicePointer(),
                m_output->GetTensorDesc(), m_output->GetGradPointer(),
                m_convDesc, m_convBwdFilterAlg, devWs, wsSize, &blendFactors[1],
                m_filter->GetFilterDesc(), m_filter->GetGradPointer()));
        }

        if (m_features->GetGradFlag()) {
            size_t bwdDataWsSize = 0;

            blendFactors[1] = m_features->IsFirstBackwardPass() ? 0 : 1;
            m_features->IncrementBackwardPass();

            checkCudaErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(
                GPUContext.GetCUDNNHandle(), m_filter->GetFilterDesc(),
                m_output->GetTensorDesc(), m_convDesc,
                m_features->GetTensorDesc(), m_convBwdDataAlg, &bwdDataWsSize));

            if (bwdDataWsSize > wsSize) {
                checkCudaErrors(cudaFree(devWs));
                checkCudaErrors(cudaMalloc(&devWs, bwdDataWsSize));
                wsSize = bwdDataWsSize;
            }

            LOG.DEBUG() << "Execuing cudnn ConvBwdData kernel "
                        << (float)wsSize / 1024000.0 << " Mb of GPU workspace.";

            checkCudaErrors(cudnnConvolutionBackwardData(
                GPUContext.GetCUDNNHandle(), &blendFactors[0],
                m_filter->GetFilterDesc(), m_filter->GetDevicePointer(),
                m_output->GetTensorDesc(), m_output->GetGradPointer(),
                m_convDesc, m_convBwdDataAlg, devWs, wsSize, &blendFactors[1],
                m_features->GetTensorDesc(), m_features->GetGradPointer()));
        }

        if (wsSize > 0) {
            checkCudaErrors(cudaFree(devWs));
        }

        cudaDeviceSynchronize();
    }

    void SetFilter(TensorPtr<FilterDataType> p) { m_filter = p; }

    void SetFeatures(TensorPtr<FeatureDataType> p) { m_features = p; }

    void SetOutput(TensorPtr<FeatureDataType> p) { m_output = p; }

    TensorBasePtr GetOutputTensor() { return m_output; }

  private:
    void Reset() {
        checkCudaErrors(cudnnCreateConvolutionDescriptor(&m_convDesc));
        m_convFwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        m_convBwdFilterAlg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        m_convBwdDataAlg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        m_scaling = {1.0, 0.0};
        m_workspaceSize = 0;
        m_workspaceBuffer = NULL;
        SetName("ConvOp");
    }

    TensorPtr<FeatureDataType> m_features;
    TensorPtr<FilterDataType> m_filter;
    TensorPtr<FilterDataType> m_bias;
    TensorPtr<FeatureDataType> m_output;
    cudnnConvolutionDescriptor_t m_convDesc;
    /**
     * These variables encode the specific
     * algorithm used by cudnn functions.
     * Each algorithm also might induce different kernels
     * depending on the compute capability and size of
     * the tensors. Right now we have the algs fixed,
     * but in the future these should be selected dynamically.
     */
    cudnnConvolutionFwdAlgo_t m_convFwdAlg;
    cudnnConvolutionBwdFilterAlgo_t m_convBwdFilterAlg;
    cudnnConvolutionBwdDataAlgo_t m_convBwdDataAlg;

    uint8_t *m_workspaceBuffer;
    size_t m_workspaceSize;

    std::array<float, 2> m_scaling;
    Stride2D m_strides;
    Pad2D m_padding;
};

template <typename T, typename V>
using ConvOpPtr = std::shared_ptr<Convolution<T, V>>;

template <typename FeatureDataType, typename FilterDataType>
TensorPtr<FeatureDataType>
MakeConvolve(TensorPtr<FeatureDataType> features, TensorPtr<FilterDataType> filter,
         Stride2D stride, Pad2D padding) {
    auto op = std::make_shared<Convolution<FeatureDataType, FilterDataType>>(
        features, filter, padding, stride);
    auto output = op->ExecuteForward();
    if (!filter->GetGradFlag() && !features->GetGradFlag())
        output->SetGradFlag(false);
    ADContext.AddOp(op);
    return output;
}
} // namespace DLFS
