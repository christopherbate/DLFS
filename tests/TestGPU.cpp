#include "QuickTestCPP.h"
#include "lib/Logging.hpp"
#include "lib/GPU.hpp"

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

int main() {
    LOG.SetMinLevel(Info);

    TestGPU();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}