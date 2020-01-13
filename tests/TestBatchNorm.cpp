#include "QuickTestCPP.h"
#include "lib/operations/BatchNorm.hpp"
#include "lib/operations/Convolution.hpp"
#include "lib/operations/SigmoidCrossEntropy.hpp"
#include "lib/tensor/Tensor.hpp"

using namespace DLFS;
using namespace std;

void TestBatchNorm() {
    TestRunner::GetRunner()->AddTest(
        "Batch Norm", "Can instatiate and execute", []() {
            auto x0 = CreateTensor({2, 2, 1, 1}, "x", 0.0f, true);
            auto x = vector<float>{1.0, 1.0, 0.0, 0.0};
            x0->CopyBufferToDevice(x);
            x0->FillConstantGrad(1.0);

            auto op = std::make_shared<BatchNormOp<float>>(x0);
            auto result = op->ExecuteForward();

            std::vector<float> resBuffer;
            result->CopyBufferToHost(resBuffer);

            auto targetValues =
                vector<float>{0.99998, 0.99998, -0.99998, -0.99998};
            unsigned idx = 0;

            for (auto num : resBuffer) {
                QTAlmostEqual(num, targetValues[idx], 1e-4);
                idx++;
            }

            result->InitGradChain();
            op->ExecuteBackward();

            // Check that the output is not distrubed.
            idx = 0;
            result->CopyBufferToHost(resBuffer);
            for (auto num : resBuffer) {
                QTAlmostEqual(num, targetValues[idx], 1e-4);
                idx++;
            }

            // Check scale param
            op->GetScaleTensor()->CopyGradBufferToHost(resBuffer);
            QTAlmostEqual(resBuffer[0], 0.0, 1e-5);

            // Check bias param
            op->GetBiasTensor()->CopyGradBufferToHost(resBuffer);
            QTAlmostEqual(resBuffer[0], 4.0, 1e-7);

            // Check input grad
            x0->CopyGradBufferToHost(resBuffer);
            for (auto num : resBuffer) {
                QTAlmostEqual(num, 0.0, 1e-7);
            }
        });

    TestRunner::GetRunner()->AddTest("Batch Norm", "Batch Norm Op", []() {
        auto w0 = vector<float>{1.62, -0.62, -0.53, -1.07};
        auto b0 = vector<float>{-0.1, 0.1};
        auto b1 = vector<float>{0.5, 0.1, 0.2};
        auto w1 = vector<float>{0.87, -2.3, 1.74, -0.76, 0.32, -0.25};
        auto y = vector<uint32_t>{2};
        auto W0 = CreateFilter(1, 2, 2, 1, "", float(0.0), true);
        auto W1 = CreateFilter(2, 3, 1, 1, "", float(0.0), true);
        auto labels = CreateTensor({1, 1, 1, 1}, "lables", uint32_t(0), false);
        auto B0 = CreateTensor({1, 1, 1, 2}, "bias-0", float(0.0), true);
        auto B1 = CreateTensor({1, 1, 1, 3}, "bias-1", float(0.0), true);
        auto x0 = CreateTensor({1, 2, 1, 1}, "x", 0.0f, false);

        vector<TensorBasePtr> params = {W0, B0, W1, B1};

        auto x = vector<float>{0.5f, 0.1f};

        x0->CopyBufferToDevice(x);
        labels->CopyBufferToDevice(y);
        B0->CopyBufferToDevice(b0);
        B1->CopyBufferToDevice(b1);
        W0->CopyBufferToDevice(w0);
        W1->CopyBufferToDevice(w1);

        int printInterval = 50;
        for (int iter = 0; iter < 500; iter++) {
            ADContext.Reset();
            auto y1 = MakeConvolve(x0, W0, Stride1, Pad0);
            auto f1 = y1 + B0;
            auto F1 = f1->ReLU();
            auto f2 = MakeConvolve(F1, W1, Stride1, Pad0) + B1;
            auto sm = f2->Softmax();
            auto loss = SigmoidCELoss(f2, labels, true);
            if (iter % printInterval == 0) {
                LOG.INFO() << loss->PrintTensor(false, false);
            }
            ADContext.CalcGradient(loss);
            ADContext.StepOptimizer(params);
        }
    });
}

int main() {
    LOG.SetMinLevel(Info);

    TestBatchNorm();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}
