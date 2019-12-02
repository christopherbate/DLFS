#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"

using namespace DLFS;
using namespace std;

void TestTensor() {
    TestRunner::GetRunner()->AddTest(
        "Tensor", "Can allocate and deallocate tensor.", []() {
            ADContext.Reset();

            Tensor<float> tensor;
            tensor.SetShape(1, 10, 10, 3);

            QTEqual(tensor.GetDevicePointer(), nullptr);

            tensor.AllocateIfNecessary();

            QTEqual(tensor.GetSizeBytes(), 4 * 10 * 10 * 3);
            QuickTest::NotEqual(tensor.GetDevicePointer(), nullptr);
        });

    TestRunner::GetRunner()->AddTest("Tensor", "Can create TensorPtr", []() {
        ADContext.Reset();

        // First api

        TensorPtr<float> tensor = std::make_shared<Tensor<float>>();
        tensor->SetShape(1, 128, 128, 3);

        QuickTest::Equal(tensor->GetDevicePointer(), nullptr);

        tensor->AllocateIfNecessary();

        QuickTest::NotEqual(tensor->GetDevicePointer(), nullptr);

        // AD Api
        TensorPtr<float> tensor2 = CreateTensor<float>(
            TensorShape({1, 128, 128, 3}), "TestTensor", 0.1, false);
        QuickTest::NotEqual(tensor2->GetDevicePointer(), nullptr);
    });

    TestRunner::GetRunner()->AddTest(
        "Tensor", "Can use constant fill initialization.", []() {
            ADContext.Reset();
            TensorPtr<float> tensor = std::make_shared<Tensor<float>>();
            tensor->SetShape(1, 100, 100, 3);
            tensor->AllocateIfNecessary();
            tensor->FillConstant(1.1f);

            vector<float> buffer(tensor->GetLinearSize());

            cudaMemcpy(buffer.data(), tensor->GetDevicePointer(),
                       tensor->GetSizeBytes(), cudaMemcpyDeviceToHost);

            for (auto val : buffer) {
                QTEqual(val, 1.1);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor", "Can convolve TensorPtr's", []() {
            ADContext.Reset();

            TensorPtr<float> features = std::make_shared<Tensor<float>>();
            features->SetShape(1, 128, 128, 3);
            features->AllocateIfNecessary();
            features->SetName("features");

            TensorPtr<float> filter = std::make_shared<Tensor<float>>();
            filter->SetFilterShape(3, 1, 3, 3);
            filter->AllocateIfNecessary();
            filter->SetName("filter");

            TensorPtr<float> result = features->Convolve(filter);
            result->SetName("output");

            /*
            Check for dangling pointers. at this point,
            we have the ref's here as well as the Op ref
            tracked by the global auto diff context.
            */

            QTEqual(features.use_count(), 2);
            QTEqual(filter.use_count(), 2);
            QTEqual(result.use_count(), 2);

            /*
            Reset the auto diff should make it so that
            we have the only ref left.
            */
            ADContext.Reset();

            QuickTest::Equal(features.use_count(), (long)1);
            QuickTest::Equal(filter.use_count(), (long)1);
            QuickTest::Equal(result.use_count(), (long)1);
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor", "Device to host copy helpers.", []() {
            ADContext.Reset();

            TensorPtr<float> features = std::make_shared<Tensor<float>>();
            features->SetGradFlag(true);
            features->SetShape(1, 12, 12, 3);
            features->SetName("features");
            features->AllocateIfNecessary();
            features->FillConstant(1.3);
            features->FillConstantGrad(0.1);

            vector<float> buffer;
            features->CopyBufferToHost(buffer);
            for (auto v : buffer) {
                QTEqual(v, 1.3);
            }

            features->CopyGradBufferToHost(buffer);
            for (auto v : buffer) {
                QTEqual(v, 0.1);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor", "Power operator overloading", []() {
            ADContext.Reset();

            TensorPtr<float> features = std::make_shared<Tensor<float>>();
            features->SetGradFlag(true);
            features->SetShape(1, 12, 12, 3);
            features->SetName("features");
            features->AllocateIfNecessary();
            features->FillConstant(3.0);
            features->FillConstantGrad(0.0);

            TensorPtr<float> out = features ^ 2.0f;

            vector<float> buffer;
            out->CopyBufferToHost(buffer);
            for (auto v : buffer) {
                QTEqual(v, 9.0);
            }
        });

    TestRunner::GetRunner()->AddTest("Tensor", "Cast", []() {
        ADContext.Reset();

        TensorShape imgShape = {1, 10, 10, 3};

        TensorPtr<uint8_t> imgTensor =
            CreateTensor<uint8_t>(imgShape, "TestImg", 1, false);

        auto convertedTensor = imgTensor->Cast<float>();

        vector<float> buffer;
        convertedTensor->CopyBufferToHost(buffer);

        for (auto v : buffer) {
            QTEqual(v, 1.0f);
        }
    });

    TestRunner::GetRunner()->AddTest(
        "Tensor", "Conv,operator+ overload combination", []() {
            ADContext.Reset();

            TensorPtr<float> features =
                CreateTensor<float>({1, 64, 64, 3}, "features", 2.0, false);

            TensorPtr<float> filter =
                CreateFilter<float>(3, 1, 3, 3, "filter", 1.0, true);

            TensorPtr<float> bias =
                CreateTensor<float>({1, 62, 62, 1}, "bias", 3.0, true);

            TensorPtr<float> result =
                features->Convolve(filter, Stride1, Pad0) + bias;

            vector<float> buffer;
            result->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 62 * 62);
            for (auto v : buffer) {
                QTEqual(v, 54.0f + 3.0);
            }
        });
}