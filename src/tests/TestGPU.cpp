#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/TensorList.hpp"
#include "operations/Convolution.hpp"
#include "GPU.hpp"

using namespace DLFS;
using namespace std;

void TestGPU()
{
    TestRunner::GetRunner()->AddTest(
        "GPU",
        "Can create cudnn handles",
        []() {
            cudnnHandle_t handle = GPUContext.GetCUDNNHandle();
            QuickTest::NotEqual(handle, NULL);
        });
}