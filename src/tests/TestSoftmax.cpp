#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "operations/Softmax.hpp"
#include "operations/SigmoidCrossEntropy.hpp"

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
        "Softmax Cross Entropy Op", "Softmax Cross Entropy Forward/Backward",
        []() {
            TensorShape shape = {8, 1, 1, 5};
            TensorShape labelShape = {8, 1, 1, 1};

            // Test with both positive and negative logits
            TensorPtr<float> logits =
                CreateTensor<float>(shape, "tensorA", 1.0);

            std::vector<uint32_t> labelBuffer = {0, 1, 2, 3, 4, 0, 1, 2};

            TensorPtr<uint32_t> labels =
                CreateTensor<uint32_t>(labelShape, "labels", 0, false);
            labels->CopyBufferToDevice(labelBuffer);

            auto ce_loss_unreduced = SoftmaxCELoss(logits, labels, false);

            std::vector<float> buffer;

            ce_loss_unreduced->CopyBufferToHost(buffer);

            QTEqual(buffer.size(), 8);

            for (auto num : buffer) {
                QTAlmostEqual(num, 1.6094, 1e-4);
            }

            auto ce_loss_reduced = SoftmaxCELoss(logits, labels, true);

            ce_loss_reduced->CopyBufferToHost(buffer);

            QTEqual(buffer.size(), 1);

            for (auto num : buffer) {
                QTAlmostEqual(num, 1.6094, 1e-4);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Sigmoid Cross Entropy Op", "Sigmoid Cross Entropy Forward/ Backward (Uneduced)",
        []() {
            ADContext.Reset();

            TensorShape shape = {4, 1, 1, 5};
            TensorShape labelShape = {4, 1, 1, 1};

            TensorPtr<float> logits = CreateTensor<float>(shape, "logits", 1.0, true);

            std::vector<uint32_t> labelBuffer = {0, 1, 2, 3};

            TensorPtr<uint32_t> labels =
                CreateTensor<uint32_t>(labelShape, "labels", 0.0, false);
            labels->CopyBufferToDevice(labelBuffer);

            auto loss = SigmoidCELoss(logits, labels, false);

            vector<float> buffer;
            loss->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 4);

            for (auto num : buffer) {
                QTAlmostEqual(num, 5.5663, 1e-4);
            }

            ADContext.CalcGradient(loss);

            logits->CopyGradBufferToHost(buffer);            

            QTEqual(buffer.size(), 20);

            for (auto num : buffer) {
                cout << num << endl;
            }
        });

     TestRunner::GetRunner()->AddTest(
        "Sigmoid Cross Entropy Op", "Sigmoid Cross Entropy Forward/ Backward (Reduced)",
        []() {
            ADContext.Reset();

            TensorShape shape = {4, 1, 1, 5};
            TensorShape labelShape = {4, 1, 1, 1};

            TensorPtr<float> logits = CreateTensor<float>(shape, "logits", 1.0, true);

            std::vector<uint32_t> labelBuffer = {0, 1, 2, 3};

            TensorPtr<uint32_t> labels =
                CreateTensor<uint32_t>(labelShape, "labels", 0.0, false);
            labels->CopyBufferToDevice(labelBuffer);

            auto loss = SigmoidCELoss(logits, labels, true);

            vector<float> buffer;
            loss->CopyBufferToHost(buffer);
            QTEqual(buffer.size(), 1);

            for (auto num : buffer) {
                QTAlmostEqual(num, 5.5663, 1e-4);
            }

            ADContext.CalcGradient(loss);
            
            logits->CopyGradBufferToHost(buffer);            
            QTEqual(buffer.size(), 20);

            for (auto num : buffer) {
                cout << num << endl;
            }
        });
}
