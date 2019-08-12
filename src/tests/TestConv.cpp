#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "tensor/AutoDiff.hpp"
#include "operations/Convolution.hpp"

using namespace DLFS;
using namespace std;

void TestConv()
{
    TestRunner::GetRunner()->AddTest(
        "Conv Op",
        "Conv op forward float",
        []() {
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

            TensorPtr<float> output = std::make_shared<Tensor<float>>();
            output->SetName("output");

            shared_ptr<Convolution<float>> convOp =
                make_shared<Convolution<float>>(array<int, 2>({0, 0}),
                                                array<int, 2>({1, 1}));

            convOp->SetFilter(filter);
            convOp->SetFeatures(features);
            convOp->SetOutput(output);

            auto outputShape = convOp->Prepare();
            output->SetShape(outputShape);

            for (auto s : outputShape)
            {
                QTEqual(s, 1);
            }

            output->AllocateIfNecessary();

            convOp->ExecuteForward();

            std::vector<float> resBuffer;
            output->CopyBufferToHost(resBuffer);

            QTEqual(resBuffer.size(), 1.0);

            float res = resBuffer[0];
            QTAlmostEqual(res, 17.6f, 1e-5);
        });

    TestRunner::GetRunner()->AddTest(
        "Conv Op",
        "Conv op backward filter float",
        []() {
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

            TensorPtr<float> output = std::make_shared<Tensor<float>>();
            output->SetName("output");
            output->SetGradFlag(true);            

            shared_ptr<Convolution<float>> convOp =
                make_shared<Convolution<float>>(Pad2d({0, 0}),
                                                Stride2d({1, 1}));
            convOp->SetFilter(filter);
            convOp->SetFeatures(features);
            convOp->SetOutput(output);

            auto outputShape = convOp->Prepare();
            output->SetShape(outputShape);
            output->AllocateIfNecessary();
            output->FillConstantGrad(0.0);

            convOp->ExecuteForward();

            std::vector<float> resBuffer;
            output->CopyBufferToHost(resBuffer);

            QTEqual(resBuffer.size(), 1);
            QTAlmostEqual(resBuffer[0], 17.6, 1e-5);

            filter->CopyGradBufferToHost(resBuffer);
            QTEqual(resBuffer.size(), 16);
            for (auto val : resBuffer)
            {
                QTEqual(val, 0.0);
            }

            output->InitGradChain();

            convOp->ExecuteBackward();

            filter->CopyGradBufferToHost(resBuffer);
            QTEqual(resBuffer.size(), 16);
            for (auto val : resBuffer)
            {
                QTAlmostEqual(val, 17.6, 1e-5);
            }
        });
}