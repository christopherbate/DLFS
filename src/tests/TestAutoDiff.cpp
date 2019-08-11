#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "tensor/AutoDiff.hpp"
#include "operations/Convolution.hpp"

#include <memory>
#include <iostream>
#include <cuda_profiler_api.h>

using namespace DLFS;
using namespace std;

void TestAutoDiff()
{
    TestRunner::GetRunner()->AddTest(
        "Convolution Operation",
        "Can execute convolution.",
        []() {
            TensorPtr tensorA = make_shared<Tensor>();
            TensorPtr tensorB = make_shared<Tensor>();
            TensorPtr tensorC = make_shared<Tensor>();

            tensorA->SetShape(8, 512, 512, 32);
            tensorA->SetPitch(4);
            tensorA->AllocateIfNecessary();

            tensorB->SetFilterShape(32, 128, 3, 3);
            tensorB->SetPitch(4);
            tensorB->AllocateIfNecessary();

            shared_ptr<Convolution> convOp = make_shared<Convolution>();
            convOp->SetFilter(tensorB);
            convOp->SetFeatures(tensorA);

            TensorPtr outputTensor = make_shared<Tensor>();
            auto filterShape = tensorB->GetShape();
            auto featShape = tensorA->GetShape();

            outputTensor->SetShape(featShape[0], featShape[1], featShape[2], filterShape[0]);
            outputTensor->SetPitch(4);
            outputTensor->AllocateIfNecessary();

            TensorShape ts = convOp->Prepare();

            cout << "Conv op prepared: " << ts[0] << " " << ts[1] << " " << ts[2] << " " << ts[3] << endl;

            cout << "Input Tensor (NHWC): " << endl;
            cout << tensorA->PrintShape() << tensorA->GetExpectedSize() << endl;

            cout << "Filter Tensor (Out Row Col In): " << endl;
            cout << tensorB->PrintShape() << tensorB->GetExpectedSize() << endl;

            cout << "Output Tensor (NHWC): " << endl;
            cout << outputTensor->PrintShape() << outputTensor->GetExpectedSize() << endl;

            convOp->SetOutput(outputTensor);
            convOp->Execute();

            cout << "Executed " << endl;

            ADContext.AddOp(convOp);
        });
}