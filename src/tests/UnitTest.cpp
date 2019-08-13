#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "Logging.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace DLFS;

void DefaultTests()
{
    TestRunner::GetRunner()->AddTest(
        "Test Failure Messages",
        "Quick Test Equal",
        []() {
            try
            {
                QuickTest::Equal(0, 1);
            }
            catch (QuickTestError &e)
            {
                return;
            }
            throw QuickTestError("QuickTest::Equal did not throw.");
        });

    TestRunner::GetRunner()->AddTest(
        "Test Failure Messages",
        "Cuda Failure",
        []() {
            try
            {
                auto test_fn = [](){
                    return (cudaError_t)1;
                };
                checkCudaErrors(test_fn());
            }
            catch (DLFSError &e)
            {
                return;
            }
            throw QuickTestError("exceptCudaErrors did not throw exception.");
        });

}

int main()
{
    LOG.SetMinLevel(Warn);
    
    DefaultTests();
    TestGPU();
    TestTensor();
    TestAutoDiff();      
    TestConv();  
    TestTensorOp();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}