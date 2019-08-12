#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "tensor/AutoDiff.hpp"
#include "operations/Convolution.hpp"

#include <memory>
#include <iostream>
#include <cuda_profiler_api.h>
#include <vector>

using namespace DLFS;
using namespace std;

void TestAutoDiff()
{
    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
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

            QTEqual(ADContext.GetOpTraceSize(), (unsigned int)1);
        });

    TestRunner::GetRunner()->AddTest(
        "AutoDiff",
        "Can calculate gradient from simple convolution, no bias for activation",
        []() {        
            ADContext.Reset();

            TensorPtr<float> features = ADContext.CreateTensor<float>();
            features->SetShape(1, 3, 3, 1);            
            features->SetName("Features");
            features->AllocateIfNecessary();
            features->FillConstant(1.0);
            

            TensorPtr<float> filter = ADContext.CreateTensor<float>();
            filter->SetFilterShape(1, 1, 3, 3);            
            filter->SetName("Filter");
            filter->AllocateIfNecessary();
            filter->FillConstant(1.0);

            TensorPtr<float> result = features->Convolve(filter);

            QuickTest::Equal(ADContext.GetOpTraceSize(), (unsigned int)1);

            std::vector<std::shared_ptr<TensorBase>> parameters;
            parameters.emplace_back(filter);
            
            ADContext.CalcGradient(result, parameters);
        });
}