#include "QuickTestCPP.h"
#include "lib/GPU.hpp"
#include "lib/tensor/Tensor.hpp"

using namespace DLFS;
using namespace std;

void TestActivation() {

    TestRunner::GetRunner()->AddTest(
        "Activation Op", "ReLU activation forward, pos", []() {
            TensorShape shape = {1, 3, 3, 1};
            TensorPtr<float> inputA =
                CreateTensor<float>(shape, "reluInput", 1.1f, true);

            auto output = inputA->ReLU();

            vector<float> buffer;

            output->CopyBufferToHost(buffer);

            for (auto &i : buffer) {
                QTEqual(i, 1.1f);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Activation Op", "ReLU activation forward, neg", []() {
            TensorShape shape = {1, 3, 3, 1};
            TensorPtr<float> inputA =
                CreateTensor<float>(shape, "reluInput", -1.1f, true);

            auto output = inputA->ReLU();

            vector<float> buffer;

            output->CopyBufferToHost(buffer);

            for (auto &i : buffer) {
                QTEqual(i, 0.0f);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Activation Op", "ReLU activation backward", []() {
            ADContext.Reset();

            TensorShape shape = {1, 1, 1, 2};
            TensorPtr<float> inputA =
                CreateTensor<float>(shape, "reluInput", 1.0f, true);
            vector<float> fill = {-1.0f, 2.0f};
            inputA->CopyBufferToDevice(fill);

            auto output = inputA->ReLU();

            vector<float> buffer(2);
            vector<float> gradBuffer(2);
            vector<float> outGradBuffer(2);

            ADContext.CalcGradient(output);

            inputA->CopyBufferToHost(buffer);
            inputA->CopyGradBufferToHost(gradBuffer);

            QTEqual(buffer.size(), 2);
            QTEqual(buffer[0], -1.0f);
            QTEqual(buffer[1], 2.0f);

            QTEqual(gradBuffer.size(), 2);
            QTEqual(gradBuffer[0], 0.0f);
            QTEqual(gradBuffer[1], 1.0f);
        });
}

int main() {
    LOG.SetMinLevel(Info);

    TestActivation();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}