#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "operations/Convolution.hpp"
#include "operations/TensorOp.hpp"

#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"

#include <memory>

using namespace DLFS;
using namespace std;

void TestTensorOp() {
    TestRunner::GetRunner()->AddTest("Tensor Op", "Tensor op add float", []() {
        TensorPtr<float> inputA = std::make_shared<Tensor<float>>();
        inputA->SetGradFlag(true);
        inputA->SetShape(1, 4, 4, 1);
        inputA->SetName("inputA");
        inputA->AllocateIfNecessary();
        inputA->FillConstant(1.0);

        TensorPtr<float> inputB = std::make_shared<Tensor<float>>();
        inputB->SetShape(1, 4, 4, 1);
        inputB->SetName("inputB");
        inputB->AllocateIfNecessary();
        inputB->FillConstant(1.0);

        TensorOpPtr<float> addOp =
            make_shared<TensorOp<float>>(PointwiseOpType::PW_ADD);
        addOp->SetScales(1.0, 1.0, 0.0);

        TensorPtr<float> out = std::make_shared<Tensor<float>>();
        out->SetGradFlag(true);
        out->SetShape(1, 4, 4, 1);
        out->SetName("add_out");
        out->AllocateIfNecessary();
        out->FillConstant(0.0);

        addOp->SetInput(inputA, 0);
        addOp->SetInput(inputB, 1);
        addOp->SetOutput(out);

        addOp->ExecuteForward();

        vector<float> buffer;

        out->CopyBufferToHost(buffer);
        for (auto val : buffer) {
            QTEqual(val, 2.0);
        }

        out->InitGradChain();

        out->CopyGradBufferToHost(buffer);
        for (auto val : buffer) {
            QTEqual(val, 1.0);
        }

        // Backward pass
        addOp->ExecuteBackward();

        buffer.clear();
        out->CopyGradBufferToHost(buffer);

        for (auto val : buffer) {
            QTEqual(val, 1.0);
        }

        buffer.clear();
        inputA->CopyGradBufferToHost(buffer);

        for (auto val : buffer) {
            QTEqual(val, 1.0);
        }
    });

    TestRunner::GetRunner()->AddTest(
        "Tensor Op", "Tensor op power float", []() {
            TensorPtr<float> inputA = std::make_shared<Tensor<float>>();
            inputA->SetShape(1, 4, 4, 1);
            inputA->SetName("inputA");
            inputA->AllocateIfNecessary();
            inputA->FillConstant(3.0);

            TensorOpPtr<float> addOp =
                make_shared<TensorOp<float>>(PointwiseOpType::PW_POW);
            addOp->SetPower(2.0);

            TensorPtr<float> out = std::make_shared<Tensor<float>>();
            out->SetShape(1, 4, 4, 1);
            out->SetName("pow_out");
            out->AllocateIfNecessary();
            out->FillConstant(0.0);

            addOp->SetInput(inputA, 0);
            addOp->SetOutput(out);

            addOp->ExecuteForward();

            vector<float> buffer;

            out->CopyBufferToHost(buffer);
            for (auto val : buffer) {
                QTEqual(val, 9.0);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor Op", "Tensor op add a vector to itself.", []() {
            ADContext.Reset();

            auto t1 = CreateTensor({1, 4, 4, 1}, "tensor1", 1.0f);

            TensorOpPtr<float> addOp =
                make_shared<TensorOp<float>>(PointwiseOpType::PW_ADD);

            addOp->SetInput(t1, 0);
            addOp->SetInput(t1, 1);
            addOp->SetOutput(t1);

            addOp->ExecuteForward();

            vector<float> buffer;
            t1->CopyBufferToHost(buffer);

            for (auto &num : buffer) {
                QTEqual(num, 2.0f);
            }
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor Op", "Tensor op add a vector's gradient to itself..", []() {
            ADContext.Reset();

            auto t1 = CreateTensor({1, 4, 4, 1}, "tensor1", 1.0f);

            t1->FillConstantGrad(0.1f);

            TensorOpPtr<float> addOp =
                make_shared<TensorOp<float>>(PointwiseOpType::PW_ADD);

            addOp->SetLHS(t1);
            addOp->SetGradRHS(t1);
            addOp->SetOutput(t1);
            addOp->SetRHSScale(-1.0f);
            addOp->ExecuteForward();            

            vector<float> buffer;

            // check that the buffer changed       
            t1->CopyBufferToHost(buffer);

            for (auto &num : buffer) {
                QTEqual(num, 0.9f);
            }

            // Check that the grad buffer is unchanged
            t1->CopyGradBufferToHost(buffer);
            for (auto &num : buffer) {
                QTEqual(num, 0.1f);
            }
        });
}
