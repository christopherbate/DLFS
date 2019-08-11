#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "tensor/AutoDiff.hpp"

using namespace DLFS;
using namespace std;

void TestTensor()
{
    TestRunner::GetRunner()->AddTest(
        "Tensor",
        "Can allocate and deallocate tensor.",
        []() {
            ADContext.Reset();
            
            Tensor<float> tensor;
            tensor.SetShape(1, 10, 10, 3);            

            QuickTest::Equal(tensor.GetPointer(), nullptr);
            QuickTest::Equal(tensor.GetPitch(), 4);

            tensor.AllocateIfNecessary();

            QuickTest::Equal(tensor.GetExpectedSize(), 4 * 10 * 10 * 3);
            QuickTest::NotEqual(tensor.GetPointer(), nullptr);
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor",
        "Can create TensorPtr",
        []() {
            ADContext.Reset();

            TensorPtr<float> tensor = std::make_shared<Tensor<float>>();
            tensor->SetShape(1, 128, 128, 3);            

            QuickTest::Equal(tensor->GetPointer(), nullptr);
            QuickTest::Equal(tensor->GetPitch(), 4);

            tensor->AllocateIfNecessary();

            QuickTest::NotEqual(tensor->GetPointer(), nullptr);
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor",
        "Can convolve TensorPtr's",
        []() {
            ADContext.Reset();

            TensorPtr<float> features = std::make_shared<Tensor<float>>();
            features->SetShape(1, 128, 128, 3);            
            features->AllocateIfNecessary();
            features->SetName("features");

            TensorPtr<float> filter = std::make_shared<Tensor<float>>();
            filter->SetFilterShape(3, 1, 3, 3);            
            filter->AllocateIfNecessary();
            filter->SetName("filter");

            TensorPtr<float> result = features->Convolve(filter);
            result->SetName("output");

            /*
            Check for dangling pointers. at this point, 
            we have the ref's here as well as the Op ref 
            tracked by the global auto diff context.            
            */
            
            QTEqual(features.use_count(), 2);
            QTEqual(filter.use_count(), 2);
            QTEqual(result.use_count(), 3);

            /*
            Reset the auto diff should make it so that 
            we have the only ref left.
            */
            ADContext.Reset();

            QuickTest::Equal(features.use_count(), 1);
            QuickTest::Equal(filter.use_count(), 1);
            QuickTest::Equal(result.use_count(), 1);
        });
}