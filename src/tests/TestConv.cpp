#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "operations/Convolution.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"

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
}