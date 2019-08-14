#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "tensor/AutoDiff.hpp"
#include "operations/Convolution.hpp"

#include <memory>
#include <iostream>
#include <cuda_profiler_api.h>
#include <vector>

using namespace DLFS;
using namespace std;

void TestAutoDiff()
{
    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Can track convolution.",
        []() {
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(4, 512, 512, 3);
            features->SetName("Features");
            features->AllocateIfNecessary();

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetFilterShape(3, 1, 3, 3);
            filter->SetName("Filter");
            filter->AllocateIfNecessary();

            TensorPtr<float> result = features->Convolve(filter);

            QTEqual(ADContext.GetOpTraceSize(), (unsigned int)1);
        });

    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Can calculate gradient from simple convolution",
        []() {
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(1, 3, 3, 1);
            features->SetName("Features");
            features->AllocateIfNecessary();
            features->FillConstant(1.0);

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetGradFlag(true);
            filter->SetFilterShape(1, 1, 3, 3);
            filter->SetName("Filter");
            filter->AllocateIfNecessary();
            filter->FillConstant(1.0);

            TensorPtr<float> result = features->Convolve(filter, {0, 0}, {1, 1});

            vector<float> buffer;
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);

            QTEqual(ADContext.GetOpTraceSize(), 1);

            std::vector<std::shared_ptr<TensorBase>> parameters;
            parameters.emplace_back(filter);

            ADContext.CalcGradient(result, parameters);

            filter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 9);
            for (auto val : buffer)
            {
                QTEqual(val, 1.0f);
            }

            // The gradient should accumulate
            // if we do not call reset, it creates additional operations.
            filter->FillConstant(2.0);
            result = features->Convolve(filter, {0, 0}, {1, 1});
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 18.0f);

            ADContext.CalcGradient(result, parameters);

            filter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 9);
            for (auto val : buffer)
            {
                QTEqual(val, 3.0f);
            }

            ADContext.Reset();

            filter->ResetBackwardPasses();
            filter->FillConstant(1.0);
            features->FillConstant(2.0);
            result = features->Convolve(filter, {0, 0}, {1, 1});
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 18.0f);

            ADContext.CalcGradient(result, parameters);

            filter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 9);
            for (auto val : buffer)
            {
                QTEqual(val, 2.0f);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Convolution + manual bias",
        []() {
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(1, 3, 3, 1);
            features->SetName("Features");
            features->AllocateIfNecessary();
            features->FillConstant(1.0);

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetGradFlag(true);
            filter->SetFilterShape(1, 1, 3, 3);
            filter->SetName("Filter");
            filter->AllocateIfNecessary();
            filter->FillConstant(1.0);

            TensorPtr<float> bias = ADContext.CreateTensor<float>();
            bias->SetShape(1, 1, 1, 1);
            bias->SetName("bias");
            bias->AllocateIfNecessary();
            bias->FillConstant(3.0);

            TensorPtr<float> result = features->Convolve(filter, {0, 0}, {1, 1});

            TensorPtr<float> result2 = result->Add(bias);

            vector<float> buffer;
            result2->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);

            QTEqual(ADContext.GetOpTraceSize(), 2);

            std::vector<std::shared_ptr<TensorBase>> parameters;
            parameters.emplace_back(filter);

            ADContext.CalcGradient(result2, parameters);

            filter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 9);
            for (auto val : buffer)
            {
                QTEqual(val, 1.0f);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Convolution + manual bias, gradient wrt conv only",
        []() {
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(1, 3, 3, 1);
            features->SetName("Features");
            features->AllocateIfNecessary();
            features->FillConstant(1.0);

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetGradFlag(true);
            filter->SetFilterShape(1, 1, 3, 3);
            filter->SetName("Filter");
            filter->AllocateIfNecessary();
            filter->FillConstant(1.0);

            TensorPtr<float> bias = ADContext.CreateTensor<float>();
            bias->SetShape(1, 1, 1, 1);
            bias->SetName("bias");
            bias->AllocateIfNecessary();
            bias->FillConstant(3.0);

            TensorPtr<float> result = features->Convolve(filter, {0, 0}, {1, 1});

            TensorPtr<float> result2 = result->Add(bias);

            vector<float> buffer;
            result2->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 12.0f);

            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 9.0f);

            QTEqual(ADContext.GetOpTraceSize(), 2);

            std::vector<std::shared_ptr<TensorBase>> parameters;
            parameters.emplace_back(filter);

            ADContext.CalcGradient(result, parameters);

            buffer.clear();
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 9.0f);

            buffer.clear();
            result->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTEqual(buffer[0], 1.0f);
        });

    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Convolution + manual bias + second conv/bias+ pow(2)",
        []() {
            ADContext.Reset();

            TensorPtr<float> features =
                ADContext.CreateTensor<float>({1, 10, 10, 3},
                                              "features", 0.5, false);

            TensorPtr<float> filter =
                ADContext.CreateFilter<float>(3, 1, 3,
                                              "filter", 1.0, true);

            TensorPtr<float> bias =
                ADContext.CreateTensor<float>({1, 8, 8, 1},
                                              "bias", 1.0, true);

            TensorPtr<float> resultFirstConv = features->Convolve(filter, {0, 0}, {1, 1});

            TensorPtr<float> secondFeatureMap = resultFirstConv + bias;

            // CHECK FIRST STAGE
            vector<float> buffer;
            secondFeatureMap->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 8 * 8);
            for (auto v : buffer)
            {
                QTEqual(v, 14.5);
            }

            TensorPtr<float> finalFilter =
                ADContext.CreateFilter<float>(1, 1, 8, "filter2", 0.1f, true);

            TensorPtr<float> finalBias =
                ADContext.CreateTensor<float>({1, 1, 1, 1}, "bias2", 3.3f, true);
            finalBias->FillConstantGrad(0.0f);

            TensorPtr<float> resultSecondConv = secondFeatureMap->Convolve(finalFilter, {0, 0}, {1, 1});            

            TensorPtr<float> result = resultSecondConv + finalBias;

            // CHECK SECOND STAGE
            buffer.clear();
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTAlmostEqual(buffer[0], 92.8f + 3.3f, 1e-4);

            auto powResult = result ^ 2.0f;

            // CHECK SECOND STAGE
            buffer.clear();
            powResult->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTAlmostEqual(buffer[0], 9235.21, 1e-2);

            std::vector<std::shared_ptr<TensorBase>> params;

            ADContext.CalcGradient(powResult, params);

            // check the gradients in reverse order.

            // Power input
            buffer.clear();
            result->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTAlmostEqual(buffer[0], 192.2, 1e-4);

            // Second bias
            buffer.clear();
            finalBias->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 1);
            QTAlmostEqual(buffer[0], 192.2, 1e-4);

            // Second conv filter
            // Gradient is incoming multiplied by corresponding point on feature map.
            // since the dimensions are equal
            buffer.clear();
            finalFilter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 64);
            for (auto v : buffer)
            {
                QTAlmostEqual(v, 192.2f * 14.5f, 1e-3);
            }

            // Second feature map
            // Gradient is incoming multipleied by corresponding point on filter map
            // since the dimensions are equal
            buffer.clear();
            secondFeatureMap->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 64);
            for (auto v : buffer)
            {
                QTAlmostEqual(v, 192.2f*0.1f, 1e-3);
            }

            // first bias
            buffer.clear();
            bias->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 64);
            for (auto v : buffer)
            {
                QTAlmostEqual(v, 192.2f*0.1f, 1e-3);
            }

            // first filter
            buffer.clear();
            filter->CopyGradBufferToHost(buffer);
            QTEqual(buffer.size(), 27);
            for (auto v : buffer)
            {
                QTAlmostEqual(v, 19.22f*32.0f, 1e-3);
            }            
        });
}