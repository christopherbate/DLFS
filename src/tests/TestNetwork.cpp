#include "QuickTestCPP.h"
#include "UnitTest.hpp"

#include "tensor/Tensor.hpp"

using namespace DLFS;
using namespace std;

void TestNetwork() {
    TestRunner::GetRunner()->AddTest(
        "Simple FC Network Test", "1-Hidden Layer", []() {
            auto w0 = vector<float>{1.62, -0.62, -0.53, -1.07};
            auto b0 = vector<float>{-0.1, 0.1};
            auto b1 = vector<float>{0.5, 0.1, 0.2};
            auto w1 = vector<float>{0.87, -2.3, 1.74, -0.76, 0.32, -0.25};
            auto y = vector<uint32_t>{2};
            auto W0 = CreateFilter(1, 2, 2, 1, "", float(0.0), true);
            auto W1 = CreateFilter(2, 3, 1, 1, "", float(0.0), true);
            auto labels =
                CreateTensor({1, 1, 1, 1}, "lables", uint32_t(0), false);
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
                auto y1 = x0->Convolve(W0, {1, 1}, {0, 0});
                auto f1 = y1 + B0;
                auto F1 = f1->ReLU();
                auto f2 = F1->Convolve(W1, {1, 1}, {0, 0}) + B1;
                auto sm = f2->Softmax();
                auto loss = f2->SigmoidCELoss(labels);
                if (iter % printInterval == 0) {
                    LOG.INFO() << loss->PrintTensor(false, false);
                }

                ADContext.CalcGradient(loss);
                ADContext.StepOptimizer(params);
            }
        });
}