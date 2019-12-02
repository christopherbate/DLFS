#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "operations/Softmax.hpp"

#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"

#include <memory>

using namespace DLFS;
using namespace std;

void TestSoftmax() {
    TestRunner::GetRunner()->AddTest("Softmax Op", "Softmax Op Forward", []() {
        TensorShape shape = {1, 1, 1, 10};
        TensorPtr<float> inputA =
            CreateTensor<float>(shape, "softmaxInputTest", 1.0);

        auto output = inputA->Softmax();

        std::vector<float> buffer;

        output->CopyBufferToHost(buffer);

        for (int i = 0; i < 10; i++) {
            QTEqual(0.1f, buffer[i]);
        }
    });

    TestRunner::GetRunner()->AddTest(
        "Sigmoid Cross Entropy Op", "Sigmoid Cross Entropy Forward/Backward",
        []() {
            TensorShape shape = {10, 1, 1, 5};
            TensorShape labelShape = {10, 1, 1, 1};

            // Test with both positive and negative logits
            TensorPtr<float> inputA =
                CreateTensor<float>(shape, "tensorA", 1.0);
            TensorPtr<float> inputB =
                CreateTensor<float>(shape, "tensorA", -1.0);

            std::vector<uint32_t> labelBuffer = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};

            TensorPtr<uint32_t> labels = CreateTensor<uint32_t>(
                labelShape, "labels", 0.0, false);
            labels->CopyBufferToDevice(labelBuffer);

            auto outputPosLogits = inputA->SigmoidCELoss(labels);
            auto outputNegLogits = inputB->SigmoidCELoss(labels);

            std::vector<float> bufferPos;
            std::vector<float> bufferNeg;

            outputNegLogits->CopyBufferToHost(bufferNeg);
            outputPosLogits->CopyBufferToHost(bufferPos);

            QTEqual(bufferNeg.size(), 50);
            QTEqual(bufferPos.size(), 50);

            float negLoss = 1.3132f;
            float posLoss = 0.3132f;

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 5; j++) {
                    if (i % 5 == j) {
                        QTAlmostEqual(bufferPos[i * 5 + j], posLoss, 1e-3);
                        QTAlmostEqual(bufferNeg[i * 5 + j], negLoss, 1e-3);
                    } else {
                        QTAlmostEqual(bufferPos[i * 5 + j], negLoss, 1e-3);
                        QTAlmostEqual(bufferNeg[i * 5 + j], posLoss, 1e-3);
                    }
                }
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Sigmoid Cross Entropy Op", "Sigmoid Cross Entropy Backward", []() {
            ADContext.Reset();

            TensorShape shape = {10, 1, 1, 5};
            TensorShape labelShape = {10, 1, 1, 1};

            // Test with both positive and negative logits
            TensorPtr<float> inputA =
                CreateTensor<float>(shape, "tensorA", 1.0);

            std::vector<uint32_t> labelBuffer = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};

            TensorPtr<uint32_t> labels = CreateTensor<uint32_t>(
                labelShape, "labels", 0.0, false);
            labels->CopyBufferToDevice(labelBuffer);

            auto loss = inputA->SigmoidCELoss(labels);

            ADContext.CalcGradient(loss);

            vector<float> posGradBuffer;
            inputA->CopyGradBufferToHost(posGradBuffer);

            float posGrad = -0.2689;
            float negGrad = 0.7311;
            cout << posGrad << negGrad << endl;
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 5; j++) {
                    if (i % 5 == j) {
                        QTAlmostEqual(posGradBuffer[i * 5 + j], posGrad, 1e-3);
                    } else {
                        QTAlmostEqual(posGradBuffer[i * 5 + j], negGrad, 1e-3);
                    }
                }
            }
        });
}
