#include "QuickTestCPP.h"
#include "lib/operations/Convolution.hpp"
#include "lib/tensor/Tensor.hpp"
#include "lib/tensor/TensorList.hpp"

using namespace DLFS;
using namespace std;

void TestConv() {
    TestRunner::GetRunner()->AddTest("Conv Op", "Conv op forward float", []() {
        TensorPtr<float> features = std::make_shared<Tensor<float>>();
        features->SetShape(1, 4, 4, 1);
        features->SetName("features");
        features->AllocateIfNecessary();
        features->FillConstant(1.0);

        QTEqual(features->GetGradPointer(), nullptr);

        TensorPtr<float> filter = std::make_shared<Tensor<float>>();
        filter->SetGradFlag(true);
        filter->SetFilterShape(1, 1, 4, 4);
        filter->SetName("filter");
        filter->AllocateIfNecessary();
        filter->FillConstant(1.1);
        filter->FillConstantGrad(0.0);

        QTNotEqual(filter->GetGradPointer(), nullptr);

        auto convOp = std::make_shared<Convolution<float, float>>(
            features, filter, array<int, 2>({0, 0}), array<int, 2>({1, 1}));

        auto output = convOp->ExecuteForward();

        std::vector<float> resBuffer;
        output->CopyBufferToHost(resBuffer);

        QTEqual(resBuffer.size(), 1.0);

        float res = resBuffer[0];
        QTAlmostEqual(res, 17.6f, 1e-5);
    });

    TestRunner::GetRunner()->AddTest(
        "Conv Op", "Conv op backward filter float", []() {
            TensorPtr<float> features = std::make_shared<Tensor<float>>();
            features->SetShape(1, 4, 4, 1);
            features->SetName("features");
            features->AllocateIfNecessary();
            features->FillConstant(1.0);

            QTEqual(features->GetGradPointer(), nullptr);

            TensorPtr<float> filter = std::make_shared<Tensor<float>>();
            filter->SetGradFlag(true);
            filter->SetFilterShape(1, 1, 4, 4);
            filter->SetName("filter");
            filter->AllocateIfNecessary();
            filter->FillConstant(1.1);
            filter->FillConstantGrad(0.0);

            QTNotEqual(filter->GetGradPointer(), nullptr);

            shared_ptr<Convolution<float, float>> convOp =
                make_shared<Convolution<float, float>>(
                    features, filter, Pad2D({0, 0}), Stride2D({1, 1}));

            auto output = convOp->ExecuteForward();

            std::vector<float> resBuffer;
            output->CopyBufferToHost(resBuffer);

            QTEqual(resBuffer.size(), 1);
            QTAlmostEqual(resBuffer[0], 17.6, 1e-5);

            filter->CopyGradBufferToHost(resBuffer);
            QTEqual(resBuffer.size(), 16);
            for (auto val : resBuffer) {
                QTEqual(val, 0.0);
            }

            output->InitGradChain();

            convOp->ExecuteBackward();

            filter->CopyGradBufferToHost(resBuffer);
            QTEqual(resBuffer.size(), 16);
            for (auto val : resBuffer) {
                QTEqual(val, 1.0);
            }
        });
    // TestRunner::GetRunner()->AddTest(
    //     "Tensor", "Can convolve TensorPtr's", []() {
    //         ADContext.Reset();

    //         TensorPtr<float> features = std::make_shared<Tensor<float>>();
    //         features->SetShape(1, 128, 128, 3);
    //         features->AllocateIfNecessary();
    //         features->SetName("features");

    //         TensorPtr<float> filter = std::make_shared<Tensor<float>>();
    //         filter->SetFilterShape(3, 1, 3, 3);
    //         filter->AllocateIfNecessary();
    //         filter->SetName("filter");

    //         TensorPtr<float> result = features->Convolve(filter);
    //         result->SetName("output");

    //         /*
    //         Check for dangling pointers. at this point,
    //         we have the ref's here as well as the Op ref
    //         tracked by the global auto diff context.
    //         */

    //         QTEqual(features.use_count(), 2);
    //         QTEqual(filter.use_count(), 2);
    //         QTEqual(result.use_count(), 2);

    //         /*
    //         Reset the auto diff should make it so that
    //         we have the only ref left.
    //         */
    //         ADContext.Reset();

    //         QuickTest::Equal(features.use_count(), (long)1);
    //         QuickTest::Equal(filter.use_count(), (long)1);
    //         QuickTest::Equal(result.use_count(), (long)1);
    //     });

    // TestRunner::GetRunner()->AddTest(
    //     "Tensor", "Conv,operator+ overload combination", []() {
    //         ADContext.Reset();

    //         TensorPtr<float> features =
    //             CreateTensor<float>({1, 64, 64, 3}, "features", 2.0, false);

    //         TensorPtr<float> filter =
    //             CreateFilter<float>(3, 1, 3, 3, "filter", 1.0, true);

    //         TensorPtr<float> bias =
    //             CreateTensor<float>({1, 62, 62, 1}, "bias", 3.0, true);

    //         TensorPtr<float> result =
    //             features->Convolve(filter, Stride1, Pad0) + bias;

    //         vector<float> buffer;
    //         result->CopyBufferToHost(buffer);
    //         QTEqual(buffer.size(), 62 * 62);
    //         for (auto v : buffer) {
    //             QTEqual(v, 54.0f + 3.0);
    //         }
    //     });
}

int main() {
    LOG.SetMinLevel(Info);

    TestConv();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}