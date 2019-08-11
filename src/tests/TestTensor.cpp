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
            Tensor tensor;
            tensor.SetShape(1, 10, 10, 3);
            tensor.SetPitch(2);

            QuickTest::Equal(tensor.GetPointer(), nullptr);

            tensor.AllocateIfNecessary();

            QuickTest::Equal(tensor.GetExpectedSize(), 1 * 10 * 10 * 3 * 2);
            QuickTest::NotEqual(tensor.GetPointer(), nullptr);
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor",
        "Can create TensorPtr",
        []() {
            TensorPtr tensor = std::make_shared<Tensor>();
            tensor->SetShape(1, 128, 128, 3);
            tensor->SetPitch(4);

            QuickTest::Equal(tensor->GetPointer(), nullptr);

            tensor->AllocateIfNecessary();

            QuickTest::NotEqual(tensor->GetPointer(), nullptr);
        });

    TestRunner::GetRunner()->AddTest(
        "Tensor",
        "Can convolve TensorPtr's",
        []() {
            TensorPtr features = std::make_shared<Tensor>();
            features->SetShape(1, 128, 128, 3);
            features->SetPitch(4);
            features->AllocateIfNecessary();

            TensorPtr filter = std::make_shared<Tensor>();
            filter->SetFilterShape(3, 1, 3, 3);
            filter->SetPitch(4);
            filter->AllocateIfNecessary();

            TensorPtr result = features->Convolve(filter);

            /*
            Check for dangling pointers. at this point, 
            we have the ref's here as well as the Op ref 
            tracked by the global auto diff context.            
            */
            
            QuickTest::Equal(features.use_count(), 2);
            QuickTest::Equal(filter.use_count(), 2);
            QuickTest::Equal(result.use_count(), 2);

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