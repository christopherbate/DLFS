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
        "Can track convolution.",
        []() {        
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(4, 512, 512, 3);            
            features->SetName("Features");
            features->AllocateIfNecessary();
            

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetFilterShape(3, 1, 3, 3);            
            filter->SetName("Filter");
            filter->AllocateIfNecessary();

            TensorPtr<float> result = features->Convolve(filter);

            QuickTest::Equal(ADContext.GetOpTraceSize(), (unsigned int)1);
        });
}