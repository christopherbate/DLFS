#include "UnitTest.hpp"
#include "QuickTestCPP.h"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"

#include <cuda_runtime.h>

#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace DLFS;

/**
 * Trains an MNIST digit classifier.
 */
void TestMNIST()
{
    TestRunner::GetRunner()->AddTest(
        "MnistTest",
        "Can load and train on MNIST",
        [](){

        }
    );
}